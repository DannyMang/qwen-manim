"""
FSDP checkpoint utilities for saving and loading model state.
"""

from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    rank: int,
    config: dict,
    is_best: bool = False,
) -> None:
    """
    Save FSDP checkpoint to disk.

    FSDP checkpointing options:
    1. FULL_STATE_DICT: Full model on rank 0 (we use this)
    2. SHARDED_STATE_DICT: Distributed checkpoint (faster, more scalable)

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer instance
        scheduler: LR scheduler instance
        epoch: Current epoch (0-indexed)
        global_step: Global training step
        checkpoint_dir: Directory to save checkpoints
        rank: Current process rank
        config: Training config dict
        is_best: Whether this is the best checkpoint so far
    """
    if rank == 0:
        print(f"\nSaving checkpoint at epoch {epoch+1}, step {global_step}...")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    dist.barrier()


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """
    Keep only the last N checkpoints to save disk space.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
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
    Load checkpoint for resuming training.

    Args:
        checkpoint_path: Path to checkpoint file
        model: FSDP-wrapped model
        optimizer: Optimizer instance
        scheduler: LR scheduler instance
        rank: Current process rank

    Returns:
        Tuple of (epoch, global_step) to resume from
    """
    if rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    load_policy = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=False,
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(checkpoint["model_state_dict"])

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
