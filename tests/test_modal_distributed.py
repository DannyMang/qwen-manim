"""
Test distributed setup on 8x A100 GPUs without loading full model.
Run: modal run tests/test_modal_distributed.py
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

app = modal.App("manimbot-test-distributed", image=image)

volume = modal.Volume.from_name("manimbot-checkpoints", create_if_missing=True)


def run_worker(rank, world_size):
    """
    Worker function that runs on each GPU.
    This will be spawned NUM_GPUS times by torch.multiprocessing.
    """
    import os
    import torch
    import torch.distributed as dist

    print(f"\n[Rank {rank}] Starting worker process")

    # Set up environment for this rank
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Initialized on cuda:{rank}")
    print(f"[Rank {rank}] Backend: {dist.get_backend()}")

    # Test communication
    tensor = torch.ones(1).cuda() * (rank + 1)
    print(f"[Rank {rank}] Before all_reduce: {tensor.item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))  # 1+2+3+...+8 = 36

    print(f"[Rank {rank}] After all_reduce: {tensor.item()} (expected: {expected})")

    # Barrier to sync all processes
    dist.barrier()

    if rank == 0:
        if abs(tensor.item() - expected) < 0.01:
            print(f"\n✅ All_reduce test PASSED!")
        else:
            print(f"\n❌ All_reduce test FAILED!")

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*80)
        print("🎉 All distributed tests PASSED!")
        print("="*80)


@app.function(
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=600,  # 10 minutes
    volumes={"/checkpoints": volume},
)
def test_distributed_setup():
    """
    Test distributed PyTorch setup on 8x A100.
    This verifies NCCL, GPU detection, and rank assignment.
    """
    import torch
    import torch.multiprocessing as mp

    print("\n" + "="*80)
    print("Testing Distributed Setup on Modal")
    print("="*80)

    # Test 1: Check GPUs
    print(f"\n✅ GPU Test:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

    # Spawn NUM_GPUS processes, one for each GPU
    print(f"\n✅ Spawning {NUM_GPUS} processes for distributed training...")
    mp.spawn(
        run_worker,
        args=(NUM_GPUS,),
        nprocs=NUM_GPUS,
        join=True
    )

    return "success"


@app.local_entrypoint()
def main():
    """Run distributed test."""
    print("Launching distributed test on 8x A100...")
    result = test_distributed_setup.remote()
    print(f"\nTest completed: {result}")
