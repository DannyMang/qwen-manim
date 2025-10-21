"""
Push trained model from Modal volume to HuggingFace Hub.
Usage: modal run src.utils.modal.push_to_hf --checkpoint-name checkpoint_best.pt --hf-repo your-username/model-name
"""

import modal

# Reuse the same image and volume from training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.20.0",
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

app = modal.App("manimbot-push-to-hf", image=image)

# Mount the same volume that contains checkpoints
volume = modal.Volume.from_name("manimbot-checkpoints", create_if_missing=False)


@app.function(
    gpu="A100-80GB:1",  # Only need 1 GPU to load and push
    timeout=7200,  # 2 hours (pushing 160GB can take time)
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # You'll need to create this
    ],
    volumes={"/checkpoints": volume},
)
def push_checkpoint_to_hf(
    checkpoint_name: str = "checkpoint_best.pt",
    hf_repo: str = None,
    private: bool = False,
):
    """
    Load checkpoint from Modal volume and push to HuggingFace Hub.

    Args:
        checkpoint_name: Name of checkpoint file (e.g., "checkpoint_best.pt")
        hf_repo: HuggingFace repo name (e.g., "username/model-name")
        private: Whether to make the repo private
    """
    import os
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if hf_repo is None:
        raise ValueError("Must provide --hf-repo argument (e.g., username/model-name)")

    print("\n" + "="*80)
    print("Pushing Model to HuggingFace Hub")
    print("="*80)
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  HF Repo: {hf_repo}")
    print(f"  Private: {private}")
    print("="*80 + "\n")

    # Verify HF token exists
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment. Create Modal secret first.")

    checkpoint_path = Path("/checkpoints") / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"✅ Found checkpoint: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")

    # Load checkpoint
    print("\n📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    config_dict = checkpoint["config"]
    model_name = config_dict["model"]["name"]

    print(f"   Original model: {model_name}")
    print(f"   Epoch: {checkpoint['epoch'] + 1}")
    print(f"   Global step: {checkpoint['global_step']}")

    # Load model config
    print("\n🔧 Loading model config...")
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Initialize model with config
    print("\n🤖 Initializing model structure...")
    model = AutoModelForCausalLM.from_config(
        model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load trained weights
    print("\n⚙️  Loading trained weights...")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Weights loaded successfully!")

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Push to HuggingFace Hub
    print(f"\n🚀 Pushing model to HuggingFace: {hf_repo}")
    print("   This may take 15-30 minutes for 160GB...")

    model.push_to_hub(
        hf_repo,
        token=hf_token,
        private=private,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB",  # Split into 5GB shards
    )

    print(f"\n✅ Model pushed successfully!")

    # Push tokenizer
    print("\n📝 Pushing tokenizer...")
    tokenizer.push_to_hub(
        hf_repo,
        token=hf_token,
        private=private,
    )

    # Create model card with training info
    print("\n📄 Creating model card...")
    model_card = f"""---
tags:
- qwen3-next
- moe
- fsdp
- manim
license: apache-2.0
---

# {hf_repo.split('/')[-1]}

Fine-tuned version of [{model_name}](https://huggingface.co/{model_name}) on Manim code generation datasets.

## Training Details

- **Base Model:** {model_name}
- **Training Framework:** PyTorch FSDP (8x A100-80GB)
- **Training Epochs:** {checkpoint['epoch'] + 1}
- **Global Steps:** {checkpoint['global_step']}
- **Mixed Precision:** bfloat16

## Datasets

- generaleoley/manim-codegen
- bespokelabs/bespoke-manim
- thanhkt/manim_code
- Edoh/manim_python

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{hf_repo}",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("{hf_repo}", trust_remote_code=True)

prompt = "Write Manim code to create a circle"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Configuration

```yaml
{config_dict['training']}
```
"""

    # Push model card
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="model",
    )

    print("\n" + "="*80)
    print("✅ PUSH COMPLETE!")
    print("="*80)
    print(f"\n🎉 Model available at: https://huggingface.co/{hf_repo}")
    print("\n")

    return f"https://huggingface.co/{hf_repo}"


@app.local_entrypoint()
def main(
    checkpoint_name: str = "checkpoint_best.pt",
    hf_repo: str = None,
    private: bool = False,
):
    """
    Push checkpoint to HuggingFace Hub.

    Example:
        modal run src.utils.modal.push_to_hf --checkpoint-name checkpoint_best.pt --hf-repo username/qwen3-80b-manim
    """
    if hf_repo is None:
        print("\n❌ Error: Must provide --hf-repo argument")
        print("\nUsage:")
        print('  modal run src.utils.modal.push_to_hf --checkpoint-name checkpoint_best.pt --hf-repo "username/model-name"')
        print("\nExample:")
        print('  modal run src.utils.modal.push_to_hf --checkpoint-name checkpoint_best.pt --hf-repo "danielung/qwen3-80b-manim"')
        return

    print(f"\n🚀 Launching push to HuggingFace...")
    url = push_checkpoint_to_hf.remote(
        checkpoint_name=checkpoint_name,
        hf_repo=hf_repo,
        private=private,
    )
    print(f"\n✅ Done! Model available at: {url}")
