# PyTorch Profiler Guide - Memory Monitoring

## 🎯 What You'll See

The profiler tracks:
- **GPU Memory Usage** (allocated, reserved, freed)
- **CPU Memory Usage**
- **Operation Timings** (forward, backward, optimizer)
- **Memory Allocations per Operation**
- **Peak Memory Usage**

## 📊 How to View Profiler Results

### **Option 1: TensorBoard (Recommended)**

#### **Step 1: Download Profiler Logs from Modal**

```bash
# After training starts, download the profiler logs
modal volume get manimbot-profiler-logs /profiler_logs ./profiler_logs
```

#### **Step 2: Launch TensorBoard**

```bash
# Install TensorBoard if you haven't
pip install tensorboard torch-tb-profiler

# Launch TensorBoard
tensorboard --logdir=./profiler_logs
```

#### **Step 3: Open in Browser**

Open: http://localhost:6006/#pytorch_profiler

#### **Step 4: Navigate to Memory View**

1. Click **"PYTORCH_PROFILER"** tab at the top
2. Select a run from the dropdown
3. Click **"Memory View"** in the left sidebar

**You'll see:**
- **Memory Timeline**: Graph showing GPU memory over time
- **Peak Memory**: Maximum memory used
- **Memory Breakdown**: Which operations use the most memory
- **Allocations**: When/where memory is allocated

### **Option 2: View in Terminal (Quick Check)**

```bash
# Download logs
modal volume get manimbot-profiler-logs /profiler_logs ./profiler_logs

# Use Python to parse the trace
python -c "
import torch
trace = torch.profiler.tensorboard_trace_handler('./profiler_logs')
print('Profiler logs saved to:', trace.path)
"
```

### **Option 3: Real-Time Monitoring (While Training)**

Add this to your training script to print memory stats:

```python
# This is already in train_one_epoch, but you can add more detail
if rank == 0 and batch_idx % 10 == 0:
    allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
    reserved = torch.cuda.memory_reserved(0) / 1e9    # GB
    print(f"  GPU 0 Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

## 🔍 Understanding the Profiler Schedule

Your config:
```yaml
profiling:
  enabled: true
  wait: 1        # Skip first 1 step (warmup)
  warmup: 2      # Next 2 steps: warmup (not profiled)
  active: 3      # Next 3 steps: ACTIVELY PROFILED
  repeat: 2      # Repeat cycle 2 times
```

**What happens:**
```
Step 1: WAIT (skip)
Step 2-3: WARMUP (prepare, not profiled)
Step 4-6: ACTIVE (🔥 profiling memory!)
Step 7: WAIT (skip)
Step 8-9: WARMUP
Step 10-12: ACTIVE (🔥 profiling again!)
Step 13+: STOPPED
```

**Total steps profiled:** 6 steps (2 cycles × 3 active steps)

## 📈 What to Look For

### **1. Peak Memory Usage**

Look for this in TensorBoard:
- **Peak Allocated**: Should be < 75GB (you have 80GB)
- **If > 78GB**: You'll get OOM errors!

### **2. Memory Leaks**

Memory should:
- ✅ **Go up during forward pass**
- ✅ **Stay high during backward**
- ✅ **Drop after optimizer step**
- ❌ **NOT keep increasing every step** (leak!)

### **3. Which Operations Use Most Memory**

TensorBoard will show:
- **Attention layers**: Usually 30-40% of memory
- **MLP layers**: 20-30%
- **Activations**: 10-20%
- **Gradients**: 25-35%

### **4. CPU Offload Effectiveness**

With `cpu_offload: true`:
- Optimizer states should show **0GB GPU usage**
- CPU memory should show **~40GB for optimizer**

## 🛠️ Interpreting Common Patterns

### **Pattern 1: Gradual Memory Increase**
```
Step 1: 45GB
Step 2: 47GB
Step 3: 49GB
Step 4: 51GB
...
```
**Issue:** Memory leak or fragmentation
**Fix:** Reduce batch size or enable more aggressive gradient checkpointing

### **Pattern 2: Sudden Spike**
```
Step 1: 45GB
Step 2: 45GB
Step 3: 78GB ⚠️  OOM!
```
**Issue:** Activation explosion or all-gather spike
**Fix:** Check if FSDP is properly sharding, enable activation checkpointing

### **Pattern 3: Stable Pattern (Good!)**
```
Step 1: 47GB
Step 2: 48GB
Step 3: 47GB
Step 4: 48GB
...
```
**Status:** ✅ Healthy training!

## 📝 Example TensorBoard Views

### **Memory Timeline**
```
80GB |
     |         ╱╲    ╱╲
60GB |    ╱╲  ╱  ╲  ╱  ╲
     |   ╱  ╲╱    ╲╱    ╲
40GB |  ╱               ╲
     | ╱                 ╲
20GB |╱___________________╲
     └─────────────────────
     Fwd  Bwd  Opt  Fwd  Bwd
```

### **Memory Breakdown**
```
Attention Layers:     35GB (45%)
MLP Layers:          20GB (26%)
Gradients:           15GB (19%)
Activations:          8GB (10%)
```

## 🚨 Troubleshooting

### **"No profiler data found"**

**Cause:** Profiler didn't run long enough
**Fix:** Train for at least 15 steps (wait + warmup + active cycles)

### **"OOM during profiling"**

**Cause:** Profiler adds ~2-3GB overhead
**Fix:** Disable profiler for actual training, only use for debugging:

```yaml
profiling:
  enabled: false  # Disable after initial test
```

### **"Can't download from Modal"**

```bash
# Check volume exists
modal volume list

# List contents
modal volume ls manimbot-profiler-logs

# If empty, profiler didn't write
# Check logs: modal run src.utils.modal.modal_app::train --detach
```

## 💡 Pro Tips

1. **Profile Early**: Run profiler for first 1-2 epochs, then disable
2. **Compare**: Profile with/without CPU offload to see savings
3. **Watch Rank 0**: Only rank 0 needs profiling (others are similar)
4. **Iterate**: If memory is tight, profile → adjust config → profile again

## 🎓 Next Steps

After viewing profiler:
1. **If memory < 70GB**: You're safe! Disable profiler for speed
2. **If memory 70-75GB**: You're okay but tight
3. **If memory > 75GB**: Enable more aggressive optimizations (see below)

### **Emergency Memory Optimizations**

```yaml
fsdp:
  cpu_offload: true                      # Already enabled ✅
  use_activation_checkpointing: true     # Already enabled ✅
  use_gradient_checkpointing: true       # Already enabled ✅

training:
  batch_size: 1                          # Already at minimum ✅
  gradient_accumulation_steps: 32        # Increase from 16 → 32

# Last resort: Reduce sequence length
model:
  max_length: 1024                       # Reduce from 2048 → 1024
```

## 📚 Resources

- [PyTorch Profiler Docs](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [TensorBoard Plugin](https://github.com/pytorch/kineto/tree/main/tb_plugin)
- [Memory Profiling Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-memory-consumption)
