"""
Test all imports without requiring GPU.
Run: python tests/test_imports.py
"""

import sys
print("Testing imports...")

try:
    import torch
    print(f"✅ torch {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ transformers: {e}")
    sys.exit(1)

try:
    import datasets
    print(f"✅ datasets")
except ImportError as e:
    print(f"❌ datasets: {e}")
    sys.exit(1)

try:
    import wandb
    print(f"✅ wandb {wandb.__version__}")
except ImportError as e:
    print(f"❌ wandb: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"✅ PyYAML")
except ImportError as e:
    print(f"❌ PyYAML: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print(f"✅ python-dotenv")
except ImportError as e:
    print(f"❌ python-dotenv: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    print(f"✅ tqdm")
except ImportError as e:
    print(f"❌ tqdm: {e}")
    sys.exit(1)

try:
    import modal
    print(f"✅ modal")
except ImportError as e:
    print(f"❌ modal: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Testing project imports...")
print("="*50)

try:
    from src.training.fsdp_config import FSDPConfig, setup_fsdp_model
    print("✅ src.training.fsdp_config")
except ImportError as e:
    print(f"❌ src.training.fsdp_config: {e}")
    sys.exit(1)

try:
    from src.training.fsdp_utils import apply_fsdp_checkpointing
    print("✅ src.training.fsdp_utils")
except ImportError as e:
    print(f"❌ src.training.fsdp_utils: {e}")
    sys.exit(1)

try:
    from src.training.callbacks import WandbLogger, TrainingMetrics
    print("✅ src.training.callbacks")
except ImportError as e:
    print(f"❌ src.training.callbacks: {e}")
    sys.exit(1)

try:
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    print("✅ src.utils.checkpoint")
except ImportError as e:
    print(f"❌ src.utils.checkpoint: {e}")
    sys.exit(1)

try:
    from src.utils.data.data_loader import get_dataloader
    print("✅ src.utils.data.data_loader")
except ImportError as e:
    print(f"❌ src.utils.data.data_loader: {e}")
    sys.exit(1)

try:
    from src.training.train import train, setup_distributed, load_config
    print("✅ src.training.train")
except ImportError as e:
    print(f"❌ src.training.train: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("🎉 All imports successful!")
print("="*50)
