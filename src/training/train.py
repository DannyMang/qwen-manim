import os
import sys
import yaml
from typing import Any
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
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

def setup_distributed():
    """
    inititializes distributed training environment
    """
    dist.init_process_group(backend="nccl")
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    local_rank=int(os.environ.get("LOCAL_RANK",0))

    torch.cuda.set_device(local_rank)

    torch.manual_seed(42+rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42+rank)

    if rank == 0:
        print(f"Initialized distributed training: world_size={world_size}")

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
    1. ALL ranks: Initialize model on meta device (0 memory)
    2. ALL ranks: Wrap with FSDP
    3. Rank 0: Load checkpoint to CPU, FSDP distributes shards to all ranks
    4. Each rank: Receives and materializes only its shard (~20GB)

    Peak memory per rank: ~30-40GB (20GB params + optimizer states)
    """
    model_name = config["model"]["name"]

    if rank == 0:
        print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = config["model"]["pad_token_id"]

    # Step 1: ALL ranks initialize model on meta device (0 memory)
    if rank == 0:
        print("All ranks: Initializing model structure on meta device...")

    config_obj = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config_obj,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    if rank == 0:
        print(f"✅ Model structure initialized (0 memory used on all ranks)")

    fsdp_cfg = FSDPConfig(
        use_mixed_precision=config["fsdp"]["use_mixed_precision"],
        cpu_offload=config["fsdp"]["cpu_offload"],
        use_activation_checkpointing=config["fsdp"]["use_activation_checkpointing"],
        use_gradient_checkpointing=config["fsdp"]["use_gradient_checkpointing"],
        sync_module_states=True,  # Will sync sharded weights from rank 0
    )

    if rank == 0:
        print("All ranks: Wrapping model with FSDP...")

    model = setup_fsdp_model(model, fsdp_cfg)

    if rank == 0:
        print("✅ Model wrapped with FSDP")

    # Step 4: Load weights - FSDP will shard and distribute
    # Only rank 0 loads from HuggingFace, FSDP broadcasts shards
    if rank == 0:
        print("Rank 0: Loading checkpoint (will be sharded across all ranks)...")

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if rank == 0:
            # Rank 0: Load full checkpoint to CPU
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
            )
            state_dict = temp_model.state_dict()
            del temp_model

            # Load into FSDP model - FSDP handles sharding
            model.load_state_dict(state_dict)
            del state_dict

            print("✅ Checkpoint loaded, distributing shards to all ranks...")

    # Wait for all ranks to receive their shards
    dist.barrier()

    if rank == 0:
        print(f"✅ Model sharded: each rank holds ~20GB")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        eps=config["training"]["adam_epsilon"],
        weight_decay=config["training"]["weight_decay"],
    )

    if rank == 0:
        print("✅ Optimizer created")

    return model, optimizer, tokenizer


def train_one_epoch(
    model: FSDP,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
    config: dict,
    logger,
    metrics: TrainingMetrics,
    rank: int,
) -> float:
    """
    Train for one epoch w/ gradient accumulation + mixed precision
    """
    model.train()
    toal_loss = 0.0
    num_batches = 0
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]
    log_interval = config["logging"]["log_every_n_steps"]
    global_step = epoch * len(dataloader)//gradient_accumulation_steps

    # do we need this?
    if rank == 0:
        pbar = tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch+1}",
            disable=False,
        )

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloder):
        start_time = time.time()
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss = loss/gradient_accumulation_steps
        loss.backwards()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step+=1

            if rank == 0 and global_step % log_interval == 0:
                avg_loss = total_loss / num_batches

                # Log to WandB
                if logger:
                    metrics.log_step_metrics(
                        logger=logger,
                        loss=avg_loss,
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                    )

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

        if rank == 0:
            pbar.update(1)

        if profiler is not None:
            profiler.step()

        if batch_idx % 100 == 0:
            dist.barrier()

    if rank == 0:
        pbar.close()

    avg_epoch_loss = total_loss/num_batches
    loss_tensor = torch.tensor([avg_epoch_loss]).cuda()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg.epoch_loss = loss_tensor.item()

    if rank == 0 and logger:
        metrics.log_epoch_metrics(
            logger=logger,
            epoch=epoch,
            avg_loss=avg_epoch_loss,
            step=global_step,
        )

    return avg_epoch_loss

def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimzer,
    scheduler,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    rank: int,
    config: dict,
) -> None:
    """
    Save FSDP checkpoint
    FSDP checkpointing options:
    1. FULL_STATE_DICT: Full model on rank 0
    2. SHARDED_STATE_DICT: Distributed checkpoint
    """

    if rank == 0:
        print(f"\nSaving checkpoint at epoch {epoch+1}, step {global_step}...")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok = True)

    save_policy = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    optim_policy = FullOptimStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        save_policy,
        optim_policy,
    ):
        model_state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)

        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optim_state_dict,
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

            if is_best:
                best_path = checkpoint_dir / "checkpoint_best.pt"
                torch.save(checkpoint, best_path)
                print(f"✅ Best checkpoint saved: {best_path}")

            keep_last_n = config["logging"].get("keep_last_n_checkpoints", 3)
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    dist,barrier()

def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """Keep only the last N checkpoints to save disk space."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )

    for checkpoint in checkpoints[:-keep_last_n]:
        checkpoint.unlink()
        print(f"Removed old checkpoint: {checkpoint}")

def load_checkpoint(
    checkpoint_path: str,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    rank: int,
) -> tuple[int, int]:
    """
    Load checkpoint for resuming training

    Returns
    (epoch, global_step)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    load_policy = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=False
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(checkpoint["model_save_dict"])

    optim_state = FSDP.optim_state_dict_to_load(
        model,
        optimizer,
        checkpoint["optimizer_state_dict"],
    )
    optimizer.load_state_dict(optim_state)

    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    if rank == 0:
        print(f"✅ Checkpoint loaded: resuming from epoch {epoch+1}, step {global_step}")

    dist.barrier()
    return epoch, global_step


def train():
    pass


if __name__ == "__main__":
    train()
