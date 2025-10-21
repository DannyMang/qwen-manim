"""
Full integration smoke test - runs actual training for 1 batch.
Tests: model loading, FSDP, dataloader, WandB, checkpointing.
Run: modal run tests/test_modal_full_smoke.py

WARNING: This loads the full 80B model! Will take 10-15 minutes.
"""

import modal

NUM_GPUS = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "wandb>=0.16.0",
        "packaging",
        "numpy",
        "tqdm",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.0",
    )
)

app = modal.App("manimbot-test-smoke", image=image)

volume = modal.Volume.from_name("manimbot-checkpoints", create_if_missing=True)


@app.function(
    gpu=modal.gpu.A100(count=NUM_GPUS),
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/checkpoints": volume},
)
def smoke_test():
    """
    Smoke test: Load model, run 1 training step, save checkpoint.
    """
    import os
    import sys
    import yaml
    import torch
    import torch.distributed as dist

    print("\n" + "="*80)
    print("SMOKE TEST: Full Training Integration")
    print("="*80)

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"\n✅ Distributed initialized: {world_size} GPUs")

    # Add project to path
    sys.path.insert(0, "/root")

    # Import training components
    from src.training.fsdp_config import FSDPConfig, setup_fsdp_model
    from src.training.callbacks import WandbLogger, TrainingMetrics
    from src.utils.data.data_loader import get_dataloader
    from src.utils.checkpoint import save_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Create minimal config for smoke test
    config = {
        "model": {
            "name": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "max_length": 512,  # Short for smoke test
            "pad_token_id": 151654,
        },
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,  # Just 1 step
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "checkpoint_dir": "/checkpoints",
        },
        "fsdp": {
            "use_mixed_precision": True,
            "cpu_offload": False,
            "use_activation_checkpointing": True,
            "use_gradient_checkpointing": True,
        },
        "logging": {
            "log_every_n_steps": 1,
            "wandb": {
                "project": "manimbot-test",
                "name": "smoke-test-run",
                "tags": ["smoke-test", "integration"],
            },
        },
    }

    # Step 1: Load model
    if rank == 0:
        print(f"\n✅ Step 1: Loading model (this may take 5-10 minutes)...")

    # All ranks initialize on meta device
    model_config = AutoConfig.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Wrap with FSDP
    fsdp_cfg = FSDPConfig(
        use_mixed_precision=config["fsdp"]["use_mixed_precision"],
        cpu_offload=config["fsdp"]["cpu_offload"],
        use_activation_checkpointing=config["fsdp"]["use_activation_checkpointing"],
        use_gradient_checkpointing=config["fsdp"]["use_gradient_checkpointing"],
        sync_module_states=True,
    )

    model = setup_fsdp_model(model, fsdp_cfg)

    # Load weights (only rank 0)
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if rank == 0:
            temp_model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
            )
            state_dict = temp_model.state_dict()
            del temp_model
            model.load_state_dict(state_dict)
            del state_dict
            print(f"  ✅ Model loaded and sharded")

    dist.barrier()

    # Step 2: Setup optimizer
    if rank == 0:
        print(f"\n✅ Step 2: Creating optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Step 3: Setup dataloader
    if rank == 0:
        print(f"\n✅ Step 3: Loading data...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = config["model"]["pad_token_id"]

    dataloader = get_dataloader(
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_length=config["model"]["max_length"],
        streaming=True,
    )

    # Step 4: Setup WandB
    logger = None
    if rank == 0:
        print(f"\n✅ Step 4: Initializing WandB...")
        logger = WandbLogger(
            project=config["logging"]["wandb"]["project"],
            name=config["logging"]["wandb"]["name"],
            config=config,
            tags=config["logging"]["wandb"]["tags"],
        )

    # Step 5: Run 1 training step
    if rank == 0:
        print(f"\n✅ Step 5: Running 1 training step...")

    model.train()
    optimizer.zero_grad()

    # Get one batch
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    labels = batch["labels"].cuda()

    # Forward
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss

    if rank == 0:
        print(f"  Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])

    # Optimizer step
    optimizer.step()

    if rank == 0:
        print(f"  ✅ Training step completed")

        # Log to WandB
        if logger:
            logger.log({"test/loss": loss.item(), "test/step": 0}, step=0)

    # Step 6: Save checkpoint
    if rank == 0:
        print(f"\n✅ Step 6: Saving checkpoint...")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,  # No scheduler for smoke test
        epoch=0,
        global_step=1,
        checkpoint_dir=config["training"]["checkpoint_dir"],
        rank=rank,
        config=config,
    )

    # Cleanup
    if rank == 0 and logger:
        logger.finish()

    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*80)
        print("🎉 SMOKE TEST PASSED!")
        print("All systems working:")
        print("  ✅ Model loading")
        print("  ✅ FSDP sharding")
        print("  ✅ Data loading")
        print("  ✅ Training step")
        print("  ✅ WandB logging")
        print("  ✅ Checkpointing")
        print("="*80)

    return "success"


@app.local_entrypoint()
def main():
    """Run smoke test."""
    print("\n⚠️  WARNING: This test loads the full 80B model!")
    print("Expected time: 10-15 minutes")
    print("Cost: ~$1-2 on Modal\n")

    import time
    time.sleep(3)  # Give user time to cancel

    print("Launching smoke test...")
    result = smoke_test.remote()
    print(f"\n✅ Smoke test completed: {result}")
