"""
Test FSDP setup with a small model (GPT2) to verify wrapping logic.
This avoids loading 80B model but tests all FSDP functionality.
Run: modal run tests/test_modal_fsdp.py
"""

import modal

NUM_GPUS = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.0",
    )
)

app = modal.App("manimbot-test-fsdp", image=image)


def run_fsdp_worker(rank, world_size):
    """
    Worker function for FSDP test (spawned per GPU).
    Tests FSDP with GPT2 model.
    """
    import os
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Set up environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize distributed
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"\n[Rank 0] Initialized {world_size} processes")
        print(f"[Rank 0] Using device: cuda:{rank}")

    # CRITICAL: Download model and tokenizer on rank 0 FIRST to avoid race conditions
    if rank == 0:
        print(f"\n✅ [Rank 0] Downloading GPT2 model and tokenizer...")
        # Force download by loading once
        _ = AutoModelForCausalLM.from_pretrained("gpt2")
        _ = AutoTokenizer.from_pretrained("gpt2")
        print(f"[Rank 0] Download complete")

    # Wait for rank 0 to finish downloading
    dist.barrier()

    if rank != 0:
        print(f"[Rank {rank}] Rank 0 download complete, proceeding...")

    # Now all ranks can safely load the model (from cache)
    if rank == 0:
        print(f"\n✅ Loading GPT2 model on all ranks...")

    # Load model on CPU first (FSDP will handle GPU placement)
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        dtype=torch.bfloat16,
    )
    # DO NOT move to GPU yet - FSDP handles this

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model loaded: {param_count:.1f}M params")

    # Wrap with FSDP
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        use_orig_params=False,  # Recommended for better performance
    )

    if rank == 0:
        print(f"  ✅ Model wrapped with FSDP")

    # Test forward pass
    if rank == 0:
        print(f"\n✅ Testing forward pass...")

    # Load tokenizer (already cached)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        "Hello world, this is a test.",
        return_tensors="pt",
        padding=True,
    )
    # Move inputs to current GPU
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Forward pass with no gradients
    with torch.no_grad():
        outputs = fsdp_model(**inputs)
        logits = outputs.logits
        dummy_loss = logits.mean()  # Dummy loss

    if rank == 0:
        print(f"  Forward pass output shape: {logits.shape}")
        print(f"  Dummy loss: {dummy_loss.item():.4f}")

    # Synchronize before backward pass
    dist.barrier()

    # Test backward pass
    if rank == 0:
        print(f"\n✅ Testing backward pass...")

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    # Forward pass with labels for loss computation
    outputs = fsdp_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    if rank == 0:
        print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients exist (FSDP may not expose all parameters)
    grad_norm = torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm=1.0)

    if rank == 0:
        print(f"  Gradient norm: {grad_norm.item():.4f}")

    # Test optimizer step
    optimizer.step()

    if rank == 0:
        print(f"  ✅ Optimizer step completed")

    # Synchronize before checkpoint
    dist.barrier()

    # Test checkpoint saving (rank 0 only)
    if rank == 0:
        print(f"\n✅ Testing checkpoint save...")

    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = fsdp_model.state_dict()
        if rank == 0:
            print(f"  State dict keys: {len(state_dict)} tensors")
            # Verify we can access a few keys
            sample_keys = list(state_dict.keys())[:3]
            print(f"  Sample keys: {sample_keys}")
            print(f"  ✅ Checkpoint save works")

    # Final barrier
    dist.barrier()

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*80)
        print("🎉 FSDP test PASSED!")
        print("="*80)


@app.function(
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=600,
)
def test_fsdp_setup():
    """
    Test FSDP wrapping with GPT2 (small model).
    Verifies: sharding, mixed precision, gradient flow.
    """
    import torch
    import torch.multiprocessing as mp

    print("\n" + "="*80)
    print("Testing FSDP Setup with GPT2")
    print("="*80)

    # Test GPU availability
    print(f"\n✅ GPU Test:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

    # Spawn NUM_GPUS processes, one for each GPU
    print(f"\n✅ Spawning {NUM_GPUS} processes for FSDP test...")

    try:
        mp.spawn(
            run_fsdp_worker,
            args=(NUM_GPUS,),
            nprocs=NUM_GPUS,
            join=True
        )
        print("\n✅ All processes completed successfully")
        return "success"
    except Exception as e:
        print(f"\n❌ Error during FSDP test: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    """Run FSDP test."""
    print("Launching FSDP test on 8x A100...")
    result = test_fsdp_setup.remote()
    print(f"\nTest completed: {result}")
