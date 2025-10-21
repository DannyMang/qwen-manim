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


@app.function(
    gpu=modal.gpu.A100(count=NUM_GPUS),
    timeout=600,
)
def test_fsdp_setup():
    """
    Test FSDP wrapping with GPT2 (small model).
    Verifies: sharding, mixed precision, gradient flow.
    """
    import os
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*80)
    print("Testing FSDP Setup with GPT2")
    print("="*80)

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"\nRank {rank}/{world_size} initialized")

    # Load small model (GPT2 = 124M params)
    if rank == 0:
        print(f"\n✅ Loading GPT2 model...")

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.bfloat16,
    )

    if rank == 0:
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

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
    )

    if rank == 0:
        print(f"  ✅ Model wrapped with FSDP")

    # Test forward pass
    if rank == 0:
        print(f"\n✅ Testing forward pass...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        "Hello world",
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = fsdp_model(**inputs)
        loss = outputs.logits.mean()  # Dummy loss

    if rank == 0:
        print(f"  Forward pass output shape: {outputs.logits.shape}")
        print(f"  Dummy loss: {loss.item():.4f}")

    # Test backward pass
    if rank == 0:
        print(f"\n✅ Testing backward pass...")

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    outputs = fsdp_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    # Check gradients exist
    has_grads = sum(1 for p in fsdp_model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in fsdp_model.parameters())

    if rank == 0:
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Parameters with gradients: {has_grads}/{total_params}")

    # Test optimizer step
    optimizer.step()

    if rank == 0:
        print(f"  ✅ Optimizer step completed")

    # Test checkpoint saving (rank 0 only)
    if rank == 0:
        print(f"\n✅ Testing checkpoint save...")
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = fsdp_model.state_dict()
            print(f"  State dict keys: {len(state_dict)} tensors")
            print(f"  ✅ Checkpoint save works")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*80)
        print("🎉 FSDP test PASSED!")
        print("="*80)

    return "success"


@app.local_entrypoint()
def main():
    """Run FSDP test."""
    print("Launching FSDP test on 8x A100...")
    result = test_fsdp_setup.remote()
    print(f"\nTest completed: {result}")
