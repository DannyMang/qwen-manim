# Testing Guide for ManimBOT Training

Incremental testing plan to verify everything works before running full training.

---

## 🚀 Quick Start

### Option 1: Run All Tests Automatically (Recommended)

```bash
python tests/run_all_tests.py
```

This will run all 5 test phases sequentially and give you a summary.

### Option 2: Run Tests Individually

Follow the phases below to run tests one by one.

---

## 🎯 Testing Phases

### Phase 1: Modal Basic Test (~ $0.01, 2 min)

**What it tests:** Modal deployment + basic imports on Modal
**Cost:** ~$0.01
**Run:**
```bash
modal run tests/test_modal_basic.py
```

**Expected output:**
```
Testing imports on Modal...
✅ torch 2.4.0
✅ transformers 4.40.0
🎉 Basic Modal test passed!
```

---

### Phase 2: Modal Distributed Test (~ $0.20, 5 min)

**What it tests:**
- 8x A100 GPU allocation
- NCCL distributed backend
- Rank assignment
- All-reduce communication
- GPU memory

**Cost:** ~$0.20
**Run:**
```bash
modal run tests/test_modal_distributed.py
```

**Expected output:**
```
✅ GPU Test:
  CUDA available: True
  GPU count: 8
  GPU 0: NVIDIA A100-SXM4-80GB

✅ Distributed Init Test:
  Rank: 0/8
  Backend: nccl

✅ Communication Test:
  Rank 0 after all_reduce: 36.0 (expected: 36.0)
  ✅ All_reduce PASSED!

🎉 All distributed tests PASSED!
```

---

### Phase 3: Modal WandB Test (~ $0.20, 3 min)

**What it tests:**
- WandB API key from Modal secrets
- WandB initialization
- Logging from rank 0 only
- Metrics tracking

**Cost:** ~$0.20
**Prerequisites:**
```bash
# Create WandB secret in Modal
modal secret create wandb-secret WANDB_API_KEY=your_key_here
```

**Run:**
```bash
modal run tests/test_modal_wandb.py
```

**Expected output:**
```
✅ WandB Test (Rank 0):
  API key found: <your_key>...
  WandB run: wandb-test-run
  WandB URL: https://wandb.ai/...
  Logged 5 test steps
  ✅ WandB run finished successfully!

🎉 WandB test PASSED!
```

**Verify:** Check your WandB dashboard for "manimbot-test" project

---

### Phase 4: Modal FSDP Test (~ $0.30, 5 min)

**What it tests:**
- FSDP model wrapping
- Mixed precision (bfloat16)
- Forward/backward pass
- Gradient computation
- Optimizer step
- Checkpoint saving

**Uses:** GPT2 (124M) instead of Qwen (80B) to save time/money

**Cost:** ~$0.30
**Run:**
```bash
modal run tests/test_modal_fsdp.py
```

**Expected output:**
```
✅ Loading GPT2 model...
  Model loaded: 124.4M params
  ✅ Model wrapped with FSDP

✅ Testing forward pass...
  Forward pass output shape: torch.Size([1, 2, 50257])
  Dummy loss: 0.0123

✅ Testing backward pass...
  Loss: 4.5678
  Parameters with gradients: 148/148
  ✅ Optimizer step completed

✅ Testing checkpoint save...
  State dict keys: 148 tensors
  ✅ Checkpoint save works

🎉 FSDP test PASSED!
```

---

### Phase 5: Full Smoke Test (~ $1-2, 10-15 min) ⚠️

**What it tests:**
- Full Qwen3-Next-80B model loading
- FSDP sharding (160GB → 20GB per GPU)
- Manim dataset loading
- Actual training step
- WandB logging
- Checkpoint saving to volume

**Cost:** ~$1-2
**Time:** 10-15 minutes
**Prerequisites:**
```bash
# Ensure WandB secret exists
modal secret list | grep wandb-secret

# Ensure checkpoint volume exists
modal volume list | grep manimbot-checkpoints
```

**Run:**
```bash
modal run tests/test_modal_full_smoke.py
```

**Expected output:**
```
✅ Step 1: Loading model (this may take 5-10 minutes)...
  ✅ Model loaded and sharded

✅ Step 2: Creating optimizer...

✅ Step 3: Loading data...

✅ Step 4: Initializing WandB...

✅ Step 5: Running 1 training step...
  Loss: 3.2145
  ✅ Training step completed

✅ Step 6: Saving checkpoint...
  ✅ Checkpoint saved: /checkpoints/checkpoint_epoch_1.pt

🎉 SMOKE TEST PASSED!
All systems working:
  ✅ Model loading
  ✅ FSDP sharding
  ✅ Data loading
  ✅ Training step
  ✅ WandB logging
  ✅ Checkpointing
```

**Verify:**
1. Check WandB dashboard: `manimbot-test/smoke-test-run`
2. Check Modal volumes: `modal volume list`

---

## 🚀 After All Tests Pass

Once all 6 phases pass, you're ready for full training:

```bash
# Full training run
modal run src.utils.modal.modal_app::train
```

---

## 📊 Estimated Costs

| Phase | Time | Cost | What it tests |
|-------|------|------|---------------|
| 1. Modal basic | 2 min | $0.01 | Modal deployment |
| 2. Distributed | 5 min | $0.20 | 8x GPU + NCCL |
| 3. WandB | 3 min | $0.20 | Logging |
| 4. FSDP | 5 min | $0.30 | Model training |
| 5. Smoke test | 15 min | $1-2 | Full integration |
| **TOTAL** | **~30 min** | **~$1.71-$2.71** | Everything |

---

## 🐛 Common Issues

### Issue: "No module named 'src'"
**Fix:**
```bash
# Make sure you're in project root
cd /Users/danielung/Desktop/projects/manimBOT
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "WANDB_API_KEY not found"
**Fix:**
```bash
# Create Modal secret
modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key_here
```

### Issue: "torch version 2.9.0 not found"
**Fix:** Already in Modal image (uses 2.4.0), but update `requirements.txt`:
```bash
# Change torch>=2.9.0 to torch>=2.4.0
```

### Issue: "Qwen3-Next not found in transformers"
**Fix:** Modal image needs transformers from main:
```python
# In modal_app.py
.pip_install("git+https://github.com/huggingface/transformers.git@main")
```

---

## ✅ Success Criteria

Before running full training, all tests should show:
- ✅ Modal deployment working
- ✅ 8x A100 GPUs allocated
- ✅ NCCL communication working (all_reduce = 36.0)
- ✅ WandB logging from rank 0
- ✅ FSDP forward/backward working
- ✅ Checkpoints saving to volume
- ✅ Training loss decreasing (smoke test only)

---

## 💡 Tips

1. **Run tests in order** - each builds on previous
2. **Check Modal dashboard** - monitor GPU usage
3. **Check WandB dashboard** - verify logging
4. **Keep smoke test output** - useful for debugging full run
5. **Cost conscious?** - Skip phase 5 (smoke test), go straight to full training after phase 4
