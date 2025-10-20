"""
FSDP utility functions for training optimization.
"""

import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def apply_fsdp_checkpointing(model: FSDP):
    """
    Apply activation checkpointing to Qwen3-Next decoder layers.

    This trades compute for memory by recomputing activations during backward pass.
    Critical for training 80B models on limited GPU memory.

    Args:
        model: FSDP-wrapped model

    Memory savings:
        - Without: ~40-50GB activations per GPU
        - With: ~10-15GB activations per GPU
        - Cost: ~20-30% slower training (recomputation overhead)
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDecoderLayer

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    # Apply checkpointing to all decoder layers
    check_fn = lambda submodule: isinstance(submodule, Qwen3NextDecoderLayer)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )
