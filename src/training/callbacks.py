import os
from typing import Any
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class WandbLogger:
    """
    WandB logger for PyTorch distributed training with FSDP support.
    Only logs from rank 0 to avoid duplicate logging.
    """

    def __init__(
        self,
        project: str = "manimbot",
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        log_model: bool = False,
        tags: list[str] | None = None,
        notes: str | None = None,
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            entity: WandB entity/team name
            name: Run name (auto-generated if None)
            config: Training config/hyperparameters to log
            log_model: Whether to save model checkpoints to WandB
            tags: List of tags for the run
            notes: Notes about the run
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.log_model = log_model
        self.tags = tags or []
        self.notes = notes

        # Determine if we should log (only rank 0)
        self.is_main_process = True
        if dist.is_initialized():
            self.is_main_process = dist.get_rank() == 0

        if self.is_main_process:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
            )
            print(f"WandB initialized: {wandb.run.name}")
        else:
            print(f"WandB skipped on rank {dist.get_rank()}")

    def log(self, metrics: dict[str, Any], step: int | None = None, commit: bool = True):
        """
        Log metrics to WandB (only on rank 0).

        Args:
            metrics: Dictionary of metric_name: value
            step: Global step (optional)
            commit: Whether to commit the log immediately
        """
        if self.is_main_process:
            wandb.log(metrics, step=step, commit=commit)

    def watch_model(self, model, log: str = "all", log_freq: int = 100):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all", or None)
            log_freq: Logging frequency
        """
        if self.is_main_process:
            wandb.watch(model, log=log, log_freq=log_freq)

    def save_checkpoint(self, checkpoint_path: str, aliases: list[str] | None = None):
        """
        Save checkpoint to WandB artifacts.

        Args:
            checkpoint_path: Path to checkpoint file
            aliases: List of aliases (e.g., ["latest", "best"])
        """
        if self.is_main_process and self.log_model:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=aliases or ["latest"])

    def finish(self):
        """Finish the WandB run."""
        if self.is_main_process:
            wandb.finish()


class TrainingMetrics:
    """
    Tracks and computes training metrics like loss, gradients, learning rate, etc.
    """

    def __init__(self, log_grad_norm: bool = True, grad_norm_freq: int = 100):
        """
        Initialize metrics tracker.

        Args:
            log_grad_norm: Whether to log gradient norms
            grad_norm_freq: Frequency to log gradient norms
        """
        self.log_grad_norm = log_grad_norm
        self.grad_norm_freq = grad_norm_freq
        self.step = 0

    def compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """
        Compute total gradient norm across all model parameters.

        Args:
            model: PyTorch model

        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_gpu_memory_stats(self) -> dict[str, float]:
        """
        Get GPU memory statistics.

        Returns:
            Dict with memory stats in GB
        """
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        return {}

    def get_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Get current learning rate from optimizer.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Current learning rate
        """
        return optimizer.param_groups[0]['lr']

    def log_step_metrics(
        self,
        logger: WandbLogger,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
    ):
        """
        Log metrics for a training step.

        Args:
            logger: WandB logger instance
            loss: Training loss
            model: Model being trained
            optimizer: Optimizer
            step: Global step number
        """
        metrics = {
            "train/loss": loss,
            "train/lr": self.get_learning_rate(optimizer),
            "train/step": step,
        }

        if self.log_grad_norm and step % self.grad_norm_freq == 0:
            grad_norm = self.compute_gradient_norm(model)
            metrics["train/grad_norm"] = grad_norm

        logger.log(metrics, step=step)

    def log_epoch_metrics(
        self,
        logger: WandbLogger,
        epoch: int,
        avg_loss: float,
        step: int,
    ):
        """
        Log metrics at the end of an epoch.

        Args:
            logger: WandB logger instance
            epoch: Epoch number
            avg_loss: Average loss for the epoch
            step: Global step number
        """
        metrics = {
            "train/epoch": epoch,
            "train/epoch_loss": avg_loss,
        }

        gpu_stats = self.get_gpu_memory_stats()
        metrics.update({f"system/{k}": v for k, v in gpu_stats.items()})

        logger.log(metrics, step=step)


def setup_wandb(
    project: str = "manimbot",
    config: dict[str, Any] | None = None,
    **kwargs
) -> WandbLogger:
    """
    Convenience function to set up WandB logging.

    Args:
        project: WandB project name
        config: Training configuration
        **kwargs: Additional arguments for WandbLogger

    Returns:
        Configured WandbLogger instance
    """
    return WandbLogger(project=project, config=config, **kwargs)
