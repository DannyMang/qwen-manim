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
    pass


def save_checkpoint(
    model: FSDP,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    rank: int,
    config: dict,
) -> None:
    pass


def load_checkpoint(
    checkpoint_path: str,
    model: FSDP,
    optimizer,
    scheduler,
    rank: int,
) -> tuple[int, int]:
    pass


def train():
    pass


if __name__ == "__main__":
    train()
