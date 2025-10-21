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


@app.function(
    gpu=modal.gpu.A100(count=NUM_GPUS),
    timeout=600,  # 10 minutes
    volumes={"/checkpoints": volume},
)
def test_distributed_setup():
    """
    Test distributed PyTorch setup on 8x A100.
    This verifies NCCL, GPU detection, and rank assignment.
    """
    import os
    import torch
    import torch.distributed as dist

    print("\n" + "="*80)
    print("Testing Distributed Setup on Modal")
    print("="*80)

    # Test 1: Check GPUs
    print(f"\n✅ GPU Test:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

    # Test 2: Initialize distributed
    print(f"\n✅ Distributed Init Test:")
    try:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"  Rank: {rank}/{world_size}")
        print(f"  Local Rank: {local_rank}")
        print(f"  Backend: {dist.get_backend()}")

        # Set device
        torch.cuda.set_device(local_rank)
        print(f"  Device set to: cuda:{local_rank}")

        # Test 3: Test communication
        print(f"\n✅ Communication Test:")
        tensor = torch.ones(1).cuda() * (rank + 1)
        print(f"  Rank {rank} before all_reduce: {tensor.item()}")

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))  # 1+2+3+...+8 = 36

        print(f"  Rank {rank} after all_reduce: {tensor.item()} (expected: {expected})")

        if rank == 0:
            if abs(tensor.item() - expected) < 0.01:
                print(f"\n✅ All_reduce PASSED!")
            else:
                print(f"\n❌ All_reduce FAILED!")

        # Test 4: Test barrier
        print(f"\n✅ Barrier Test:")
        dist.barrier()
        if rank == 0:
            print("  All ranks synchronized!")

        # Cleanup
        dist.destroy_process_group()

        if rank == 0:
            print("\n" + "="*80)
            print("🎉 All distributed tests PASSED!")
            print("="*80)

        return "success"

    except Exception as e:
        print(f"\n❌ Distributed test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    """Run distributed test."""
    print("Launching distributed test on 8x A100...")
    result = test_distributed_setup.remote()
    print(f"\nTest completed: {result}")
