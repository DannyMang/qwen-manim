import functools
import torch
import torch.distributed as dist
from dataclasses import dataclass
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from src.training.fsdp_utils import apply_fsdp_checkpointing

@dataclass
class FSDPConfig:
    """
    FSDP Training config for Qwen3-Next-80B
    """
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #zero-3

    use_mixed_precision: bool = True
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    buffer_dtype: torch.dtype = torch.bfloat16

    cpu_offload:bool = False # enable later if OOM?
    use_activation_checkpointing: bool = True
    use_gradient_checkpointing: bool = True

    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
    forward_prefetch:bool = True
    limit_all_gathers: bool = True

    sync_module_states: bool = True
    use_orig_params: bool = True


def get_qwen3next_auto_wrap_policy():
    """
    auto wrap policy for Qwen3-next decoder layers

    wraps each of 48 Qwen3NextDecoderLayer instances,
    each layer either linear_attention or full_attention + MoE block w/ 512 experts (10 active)
    """
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import(
            Qwen3NextDecoderLayer
        )
    except ImportError:
        raise ImportError(
            "Qwen3-Next not found, install via pip install git+https://github.com/huggingface/transformers.git@main"
        )

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3NextDecoderLayer},
    )


def get_mixed_precision_policy(config: FSDPConfig) -> MixedPrecision | None:
    """Create mixed precision policy for FSDP."""
    if not config.use_mixed_precision:
        return None

    return MixedPrecision(
        param_dtype=config.param_dtype,
        reduce_dtype=config.reduce_dtype,
        buffer_dtype=config.buffer_dtype,
        cast_forward_inputs=True,
    )

def setup_fsdp_model(model: torch.nn.Module, config: FSDPConfig = None) -> FSDP:
    """
    Wrap Qwen3 Next model with FSDP
    """
    if config is None:
        config=FSDPConfig()

    if config.use_gradient_checkpointing and hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    fsdp_model = FSDP(
        model,
        sharding_strategy=config.sharding_strategy,
        auto_wrap_policy=get_qwen3next_auto_wrap_policy(),
        mixed_precision=get_mixed_precision_policy(config),
        cpu_offload=CPUOffload(offload_params=True) if config.cpu_offload else None,
        backward_prefetch=config.backward_prefetch,
        forward_prefetch=config.forward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
        sync_module_states=config.sync_module_states,
        use_orig_params=config.use_orig_params,
        device_id=torch.cuda.current_device(),
    )

    if config.use_activation_checkpointing:
        apply_fsdp_checkpointing(fsdp_model)

    return fsdp_model
