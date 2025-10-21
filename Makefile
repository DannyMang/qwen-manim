# Makefile for ManimBOT Training

.PHONY: help test test-all test-local test-modal-basic test-distributed test-wandb test-fsdp train deploy push-to-hf

help:
	@echo "ManimBOT Training Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all          - Run all tests sequentially"
	@echo "  make test-modal-basic  - Phase 1: Modal basic (\$$0.01)"
	@echo "  make test-distributed  - Phase 2: Distributed setup (\$$0.20)"
	@echo "  make test-wandb        - Phase 3: WandB logging (\$$0.20)"
	@echo "  make test-fsdp         - Phase 4: FSDP with GPT2 (\$$0.30)"
	@echo ""
	@echo "Training:"
	@echo "  make train             - Run full training on Modal"
	@echo "  make deploy            - Deploy Modal app"
	@echo "  make push-to-hf        - Push trained model to HuggingFace Hub"
	@echo ""
	@echo "Setup:"
	@echo "  make install           - Install dependencies"
	@echo "  make wandb-secret      - Create WandB secret in Modal"
	@echo "  make hf-secret         - Create HuggingFace secret in Modal"

# Testing
test-all:
	@python tests/run_all_tests.py

test: test-all

test-local:
	@echo "Running Phase 1: Local imports..."
	@python tests/test_imports.py

test-modal-basic:
	@echo "Running Phase 2: Modal basic..."
	@modal run tests/test_modal_basic.py

test-distributed:
	@echo "Running Phase 3: Distributed setup..."
	@modal run tests/test_modal_distributed.py

test-wandb:
	@echo "Running Phase 4: WandB logging..."
	@echo "⚠️  Requires WandB secret. Run 'make wandb-secret' first."
	@modal run tests/test_modal_wandb.py

test-fsdp:
	@echo "Running Phase 4: FSDP..."
	@modal run tests/test_modal_fsdp.py

# Training
train:
	@echo "⚠️  WARNING: Full training run!"
	@echo "Cost: ~\$$50-100+, Time: Hours"
	@read -p "Continue? (y/n): " confirm && [ $$confirm = "y" ] || exit 1
	@modal run src.utils.modal.modal_app::train

deploy:
	@echo "Deploying Modal app..."
	@modal deploy src/utils/modal/modal_app.py

push-to-hf:
	@echo "Pushing model to HuggingFace Hub..."
	@echo "⚠️  Requires HuggingFace token. Run 'make hf-secret' first."
	@read -p "HuggingFace repo (e.g., username/model-name): " repo && \
	 read -p "Checkpoint name [checkpoint_best.pt]: " checkpoint && \
	 checkpoint=$${checkpoint:-checkpoint_best.pt} && \
	 modal run src.utils.modal.push_to_hf --checkpoint-name $$checkpoint --hf-repo $$repo

# Setup
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

wandb-secret:
	@echo "Creating WandB secret in Modal..."
	@echo "Enter your WandB API key:"
	@read -p "API Key: " key && modal secret create wandb-secret WANDB_API_KEY=$$key
	@echo "✅ WandB secret created!"

hf-secret:
	@echo "Creating HuggingFace secret in Modal..."
	@echo "Get your token from: https://huggingface.co/settings/tokens"
	@echo "Enter your HuggingFace token:"
	@read -p "HF Token: " token && modal secret create huggingface-secret HF_TOKEN=$$token
	@echo "✅ HuggingFace secret created!"

# Clean
clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"
