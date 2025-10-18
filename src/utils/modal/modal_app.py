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
    pass
