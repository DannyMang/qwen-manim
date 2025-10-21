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
    .add_local_dir(
        local_path="src",
        remote_path="/root/src"
    )
    .add_local_dir(
        local_path="config",
        remote_path="/root/config"
    )
)

app = modal.App("manimbot-training", image=image)

# Create volumes for checkpoints and profiler logs
checkpoint_volume = modal.Volume.from_name("manimbot-checkpoints", create_if_missing=True)
profiler_volume = modal.Volume.from_name("manimbot-profiler-logs", create_if_missing=True)


def run_training_worker(rank, world_size, config_path="/root/config/training_config.yaml"):
    """
    Worker function for training (spawned per GPU).
    """
    import os
    import sys

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)


    sys.path.insert(0, "/root")

    from src.training.train import train as train_fn

    train_fn(config_path=config_path)


@app.function(
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/profiler_logs": profiler_volume,
    },
    retries=3,
)
def train():
    """
    Main training function for Modal.
    Runs FSDP training on 8x A100 GPUs using torch.multiprocessing.
    """
    import torch
    import torch.multiprocessing as mp

    print("\n" + "="*80)
    print("Starting FSDP Training on Modal")
    print("="*80)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    print("="*80 + "\n")

    print(f"Launching training on {NUM_GPUS}x A100-80GB GPUs...")

    # Spawn NUM_GPUS processes, one for each GPU
    try:
        mp.spawn(
            run_training_worker,
            args=(NUM_GPUS,),
            nprocs=NUM_GPUS,
            join=True
        )
        print("\n✅ Training completed successfully!")
        return "success"
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    """Run training from command line."""
    print("\n🚀 Starting training job on Modal...")
    result = train.remote()
    print(f"\n✅ Training job completed: {result}")
