import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Optional
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)

load_dotenv()

from src.training.fsdp_config import FSDPConfig, setup_fsdp_model
from src.training.callbacks import WandbLogger, TrainingMetrics
from src.utils.data.data_loader import get_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint

def setup_distributed():
    """
    Initializes distributed training environment.
    Assumes RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT are set by torch.multiprocessing.spawn or torchrun.
    """
    # Get rank info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Initialize process group with explicit rank/world_size (like in tests)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)

    torch.manual_seed(42+rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42+rank)

    if rank == 0:
        print(f"[Rank 0] Initialized distributed training: world_size={world_size}")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def load_config(config_path: str = "config/training_config.yaml") -> dict[str, Any]:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_optimizer(config: dict, rank: int):
    """
    Load Qwen3-Next model and wrap with FSDP with efficient sharding.

    Sharded loading approach:
    1. Rank 0: Download model config and tokenizer to cache (prevents race conditions)
    2. ALL ranks: Initialize model on meta device (0 memory)
    3. ALL ranks: Wrap with FSDP
    4. Rank 0: Load checkpoint to CPU, FSDP distributes shards to all ranks
    5. Each rank: Receives and materializes only its shard (~20GB)

    Peak memory per rank: ~30-40GB (20GB params + optimizer states)
    """
    model_name = config["model"]["name"]

    if rank == 0:
        print(f"[Rank 0] Loading model: {model_name}")

    # Download config and tokenizer on rank 0 FIRST to avoid race conditions
    if rank == 0:
        print(f"[Rank 0] Downloading model config and tokenizer...")
        # Force download by loading once
        _ = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        print(f"[Rank 0] Download complete")

    # Wait for rank 0 to finish downloading
    dist.barrier()

    if rank != 0:
        print(f"[Rank {rank}] Rank 0 download complete, proceeding...")

    # Now all ranks can safely load from cache
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = config["model"]["pad_token_id"]

    # Step 1: ALL ranks initialize model on meta device (0 memory)
    if rank == 0:
        print(f"\n[Rank 0] Initializing model structure on meta device...")

    config_obj = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config_obj,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

    if rank == 0:
        print(f"[Rank 0] ✅ Model structure initialized (0 memory used on all ranks)")

    fsdp_cfg = FSDPConfig(
        use_mixed_precision=config["fsdp"]["use_mixed_precision"],
        cpu_offload=config["fsdp"]["cpu_offload"],
        use_activation_checkpointing=config["fsdp"]["use_activation_checkpointing"],
        use_gradient_checkpointing=config["fsdp"]["use_gradient_checkpointing"],
        sync_module_states=True,  # Will sync sharded weights from rank 0
    )

    if rank == 0:
        print("[Rank 0] All ranks: Wrapping model with FSDP...")

    model = setup_fsdp_model(model, fsdp_cfg)

    if rank == 0:
        print("[Rank 0] ✅ Model wrapped with FSDP")

    # Step 4: Load weights - FSDP will shard and distribute
    # Only rank 0 loads from HuggingFace, FSDP broadcasts shards
    if rank == 0:
        print("[Rank 0] Loading checkpoint (will be sharded across all ranks)...")

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if rank == 0:
            # Rank 0: Load full checkpoint to CPU
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
            )
            state_dict = temp_model.state_dict()
            del temp_model

            # Load into FSDP model - FSDP handles sharding
            model.load_state_dict(state_dict)
            del state_dict

            print("[Rank 0] ✅ Checkpoint loaded, distributing shards to all ranks...")

    # Wait for all ranks to receive their shards
    dist.barrier()

    if rank == 0:
        print(f"[Rank 0] ✅ Model sharded: each rank holds ~20GB")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        eps=config["training"]["adam_epsilon"],
        weight_decay=config["training"]["weight_decay"],
    )

    if rank == 0:
        print("[Rank 0] ✅ Optimizer created")

    return model, optimizer, tokenizer


def train_one_epoch(
    model: FSDP,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
    config: dict,
    logger: Optional[WandbLogger],
    metrics: TrainingMetrics,
    rank: int,
    profiler: Optional = None,
) -> float:
    """
    Train for one epoch w/ gradient accumulation + mixed precision.

    Args:
        model: FSDP-wrapped model
        dataloader: Training dataloader
        optimizer: Optimizer instance
        scheduler: LR scheduler
        epoch: Current epoch (0-indexed)
        config: Training configuration
        logger: WandB logger (optional)
        metrics: Metrics tracker
        rank: Process rank
        profiler: PyTorch profiler (optional)

    Returns:
        Average epoch loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]
    log_interval = config["logging"]["log_every_n_steps"]
    global_step = epoch * len(dataloader) // gradient_accumulation_steps

    # Progress bar on rank 0 only
    if rank == 0:
        pbar = tqdm(
            total=len(dataloader) // gradient_accumulation_steps,
            desc=f"Epoch {epoch+1}",
            disable=False,
        )

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to GPU
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Optimizer step after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Logging (rank 0 only)
            if rank == 0:
                avg_loss = total_loss / num_batches

                if global_step % log_interval == 0:
                    # Log to WandB
                    if logger:
                        metrics.log_step_metrics(
                            logger=logger,
                            loss=avg_loss,
                            model=model,
                            optimizer=optimizer,
                            step=global_step,
                        )

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })
                pbar.update(1)

        # Profiler step (if enabled)
        if profiler is not None:
            profiler.step()

        # Periodic barrier to keep ranks synchronized
        if batch_idx % 100 == 0:
            dist.barrier()

    if rank == 0:
        pbar.close()

    # Calculate epoch average loss across all ranks
    avg_epoch_loss = total_loss / num_batches
    loss_tensor = torch.tensor([avg_epoch_loss]).cuda()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_epoch_loss = loss_tensor.item()

    # Log epoch metrics (rank 0 only)
    if rank == 0 and logger:
        metrics.log_epoch_metrics(
            logger=logger,
            epoch=epoch,
            avg_loss=avg_epoch_loss,
            step=global_step,
        )

    return avg_epoch_loss


def train(config_path: str = "config/training_config.yaml"):
    """
    Main training function with FSDP, profiling, and checkpointing.

    This orchestrates:
    1. Distributed setup
    2. Model/optimizer initialization
    3. Data loading
    4. Training loop with epochs
    5. Checkpointing
    6. Profiling (optional)
    7. Cleanup
    """
    rank, world_size, local_rank = setup_distributed()

    try:
        config = load_config(config_path)

        if rank == 0:
            print("\n" + "="*80)
            print("Starting FSDP Training - Qwen3-Next-80B-A3B")
            print("="*80)
            print(f"World size: {world_size}")
            print(f"Model: {config['model']['name']}")
            print(f"Epochs: {config['training']['num_epochs']}")
            print(f"Batch size per GPU: {config['training']['batch_size']}")
            print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
            effective_batch = (
                config['training']['batch_size'] *
                world_size *
                config['training']['gradient_accumulation_steps']
            )
            print(f"Effective batch size: {effective_batch}")
            print("="*80 + "\n")

        model, optimizer, tokenizer = setup_model_and_optimizer(config, rank)

        if rank == 0:
            print("[Rank 0] Setting up data loader...")

        dataloader = get_dataloader(
            tokenizer=tokenizer,
            batch_size=config["training"]["batch_size"],
            max_length=config["model"]["max_length"],
            streaming=True,
            num_workers=0,  # Must be 0 for streaming
        )

        if rank == 0:
            print(f"[Rank 0] ✅ Data loader ready")

        num_training_steps = (
            len(dataloader) //
            config["training"]["gradient_accumulation_steps"] *
            config["training"]["num_epochs"]
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["training"]["warmup_steps"],
            num_training_steps=num_training_steps,
        )

        if rank == 0:
            print(f"[Rank 0] ✅ Scheduler ready ({num_training_steps} steps)")

        # Setup WandB logging (rank 0 only)
        logger = None
        if rank == 0:
            logger = WandbLogger(
                project=config["logging"]["wandb"]["project"],
                name=config["logging"]["wandb"]["name"],
                config=config,
                tags=config["logging"]["wandb"].get("tags", []),
            )
            print("[Rank 0] ✅ WandB initialized")

        metrics = TrainingMetrics(
            log_grad_norm=True,
            grad_norm_freq=config["logging"].get("grad_norm_freq", 100),
        )

        profiler = None
        if config.get("profiling", {}).get("enabled", False):
            if rank == 0:
                print("[Rank 0] Setting up PyTorch profiler...")

            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=1,
                    warmup=2,
                    active=3,
                    repeat=2,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    './profiler_logs'
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()

            if rank == 0:
                print("[Rank 0] ✅ Profiler enabled")

        # Load checkpoint if resuming
        start_epoch = 0
        resume_from = config.get("training", {}).get("resume_from_checkpoint", None)
        if resume_from and Path(resume_from).exists():
            start_epoch, _ = load_checkpoint(
                resume_from,
                model,
                optimizer,
                scheduler,
                rank,
            )

        if rank == 0:
            print("\n" + "="*80)
            print("Starting training loop")
            print("="*80 + "\n")

        best_loss = float('inf')

        for epoch in range(start_epoch, config["training"]["num_epochs"]):
            if rank == 0:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
                print(f"{'='*80}\n")

            # Train one epoch
            avg_loss = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                logger=logger,
                metrics=metrics,
                rank=rank,
                profiler=profiler,
            )

            if rank == 0:
                print(f"\n✅ Epoch {epoch+1} complete - Average loss: {avg_loss:.4f}")

            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            save_every_n = config["logging"].get("save_every_n_epochs", 1)
            if (epoch + 1) % save_every_n == 0:
                global_step = (epoch + 1) * len(dataloader) // config["training"]["gradient_accumulation_steps"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    checkpoint_dir=config["training"].get("checkpoint_dir", "./checkpoints"),
                    rank=rank,
                    config=config,
                    is_best=is_best,
                )

        # Stop profiler
        if profiler is not None:
            profiler.stop()
            if rank == 0:
                print("[Rank 0] ✅ Profiler stopped")

        # Finish WandB
        if rank == 0 and logger:
            logger.finish()
            print("[Rank 0] ✅ WandB finished")

        if rank == 0:
            print("\n" + "="*80)
            print("✅ Training complete!")
            print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ [Rank {rank}] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        cleanup_distributed()
        if rank == 0:
            print("[Rank 0] Distributed training cleaned up")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    train(config_path)
