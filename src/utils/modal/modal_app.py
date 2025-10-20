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

app = modal.App("manimbot-training", image=image)

# Create volume for checkpoints
volume = modal.Volume.from_name("manimbot-checkpoints", create_if_missing=True)

@app.function(
    gpu=modal.gpu.A100(count=NUM_GPUS),
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/checkpoints": volume},
    retries=3,
)
def train():
    """
    Main training function for Modal.
    Runs FSDP training on 8x A100 GPUs.
    """
    import os
    import sys

    # Add project root to path
    sys.path.insert(0, "/root")

    # Import and run training
    from src.training.train import train as train_fn

    # Run training with config
    train_fn(config_path="/root/config/training_config.yaml")
