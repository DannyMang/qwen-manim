"""
Test WandB logging on Modal with distributed setup.
Run: modal run tests/test_modal_wandb.py
"""

import modal

NUM_GPUS = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "wandb>=0.16.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.0",
    )
)

app = modal.App("manimbot-test-wandb", image=image)


def run_wandb_worker(rank, world_size):
    """
    Worker function for WandB test (spawned per GPU).
    """
    import os
    import torch
    import torch.distributed as dist
    import wandb

    # Set up environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize distributed
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"\n[Rank {rank}] Initialized on cuda:{rank}")

    # Test WandB (only rank 0)
    if rank == 0:
        print(f"\n✅ WandB Test (Rank 0):")

        # Check for API key
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            print(f"  API key found: {api_key[:10]}...")
        else:
            print(f"  ❌ WARNING: No WANDB_API_KEY found!")

        # Initialize WandB
        wandb.init(
            project="manimbot-test",
            name="wandb-test-run",
            config={
                "test": "wandb_logging",
                "gpus": NUM_GPUS,
                "world_size": world_size,
            },
            tags=["test", "wandb"],
        )

        print(f"  WandB run: {wandb.run.name}")
        print(f"  WandB URL: {wandb.run.url}")

        # Log some test metrics
        for step in range(5):
            wandb.log({
                "test/step": step,
                "test/loss": 1.0 / (step + 1),
                "test/rank": rank,
            }, step=step)

        print(f"  Logged 5 test steps")

        # Finish
        wandb.finish()
        print(f"  ✅ WandB run finished successfully!")

    else:
        print(f"\n[Rank {rank}] Skipping WandB (not rank 0)")

    # Barrier
    dist.barrier()

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*80)
        print("🎉 WandB test PASSED!")
        print("="*80)


@app.function(
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def test_wandb_logging():
    """
    Test WandB logging with distributed training (only rank 0 logs).
    """
    import torch
    import torch.multiprocessing as mp

    print("\n" + "="*80)
    print("Testing WandB Logging on Modal")
    print("="*80)

    # Test GPU availability
    print(f"\n✅ GPU Test:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")

    # Spawn NUM_GPUS processes, one for each GPU
    print(f"\n✅ Spawning {NUM_GPUS} processes for distributed WandB test...")
    mp.spawn(
        run_wandb_worker,
        args=(NUM_GPUS,),
        nprocs=NUM_GPUS,
        join=True
    )

    return "success"


@app.local_entrypoint()
def main():
    """Run WandB test."""
    print("Launching WandB test on 8x A100...")
    result = test_wandb_logging.remote()
    print(f"\nTest completed: {result}")
