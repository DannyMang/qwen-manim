# Makefile for ManimBOT Training

.PHONY: help test test-all test-local test-modal-basic test-distributed test-wandb test-fsdp test-smoke train deploy

help:
	@echo "ManimBOT Training Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all          - Run all tests sequentially"
	@echo "  make test-modal-basic  - Phase 1: Modal basic (\$$0.01)"
	@echo "  make test-distributed  - Phase 2: Distributed setup (\$$0.20)"
	@echo "  make test-wandb        - Phase 3: WandB logging (\$$0.20)"
	@echo "  make test-fsdp         - Phase 4: FSDP with GPT2 (\$$0.30)"
	@echo "  make test-smoke        - Phase 5: Full smoke test (\$$1-2)"
	@echo ""
	@echo "Training:"
	@echo "  make train             - Run full training on Modal"
	@echo "  make deploy            - Deploy Modal app"
	@echo ""
	@echo "Setup:"
	@echo "  make install           - Install dependencies"
	@echo "  make wandb-secret      - Create WandB secret in Modal"

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
	@echo "Running Phase 5: FSDP..."
	@modal run tests/test_modal_fsdp.py

test-smoke:
	@echo "Running Phase 6: Full smoke test..."
	@echo "⚠️  WARNING: Loads full 80B model. Cost: ~\$$1-2, Time: 15 min"
	@read -p "Continue? (y/n): " confirm && [ $$confirm = "y" ] || exit 1
	@modal run tests/test_modal_full_smoke.py

# Training
train:
	@echo "⚠️  WARNING: Full training run!"
	@echo "Cost: ~\$$50-100+, Time: Hours"
	@read -p "Continue? (y/n): " confirm && [ $$confirm = "y" ] || exit 1
	@modal run src.utils.modal.modal_app::train

deploy:
	@echo "Deploying Modal app..."
	@modal deploy src/utils/modal/modal_app.py

# Setup
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

wandb-secret:
	@echo "Creating WandB secret in Modal..."
	@echo "Enter your WandB API key:"
	@read -p "API Key: " key && modal secret create wandb-secret WANDB_API_KEY=$$key
	@echo "✅ WandB secret created!"

# Clean
clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"
