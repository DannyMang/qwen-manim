# Testing Guide for ManimBOT Training

Incremental testing plan to verify everything works before running full training.

---

## 🚀 Quick Start

### Option 1: Run All Tests Automatically (Recommended)

```bash
python tests/run_all_tests.py
```

This will run all 4 test phases sequentially and give you a summary.

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

## 🚀 After All Tests Pass

Once all 4 phases pass, you're ready for full training:

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
| **TOTAL** | **~15 min** | **~$0.71** | Everything |

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

---

## 💡 Tips

1. **Run tests in order** - each builds on previous
2. **Check Modal dashboard** - monitor GPU usage
3. **Check WandB dashboard** - verify logging
4. **All tests pass?** - You're ready for full training!
