---
name: train-speculator
description: Run a smoke training for a speculator model to validate the implementation learns correctly.
---

# Smoke Training for a Speculator Implementation

You are running a short smoke training to validate that a speculator implementation is correct and can learn. The user provides the algorithm name (speculator type).

## Step 1: Pre-flight Checks

### Check GPUs
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```
Count available GPUs and note memory. This determines the verifier model size and parallelism.

### Check the implementation exists
```bash
cd /workspace/speculators
/workspace/speculators/.venv/bin/python -c "from speculators.model import SpeculatorModel; print(list(SpeculatorModel.registry.keys()))"
```
Verify the algorithm name appears in the registry. If not, tell the user to run `/implement-speculator` first.

### Pick a verifier model
Based on GPU count and memory:
- **1-2 GPUs, 40-80GB**: Use `Qwen/Qwen3-0.6B` (small, fast, good for validation)
- **4+ GPUs, 80GB each**: Use `Qwen/Qwen3-8B` or `meta-llama/Llama-3.1-8B`

Check if any models are already cached locally:
```bash
ls /root/.cache/huggingface/hub/ 2>/dev/null | grep -E 'qwen|llama' | head -5
find /workspace -maxdepth 2 -name "config.json" -path "*/models*" 2>/dev/null | head -5
```

### Determine training configuration
Read `scripts/train.py` to find what arguments the new speculator type needs.
Read existing training examples in `examples/train/` for reference.

## Step 2: Prepare Data

Run data preparation with a small dataset:
```bash
cd /workspace/speculators
/workspace/speculators/.venv/bin/python scripts/prepare_data.py \
    --model <VERIFIER_MODEL> \
    --data sharegpt \
    --output ./output/smoke_<algo_name> \
    --max-samples 500 \
    --seq-length 2048
```

If `sharegpt` is not available, try `ultrachat` or check what datasets are cached.

## Step 3: Run Smoke Training

### Determine GPU split
For online training (needs vLLM server + training GPUs):
- If 4+ GPUs: 2 for vLLM, remaining for training
- If 2 GPUs: Use offline mode (pre-generate hidden states) or 1 GPU each

### Option A: Offline training (simpler, fewer GPUs)

Run training directly without a vLLM server. The training script generates hidden states on-the-fly from the verifier model loaded on the training GPUs:
```bash
cd /workspace/speculators
CUDA_VISIBLE_DEVICES=0,1 /workspace/speculators/.venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=<NUM_GPUS> \
    scripts/train.py \
    --verifier-name-or-path <VERIFIER_MODEL> \
    --speculator-type <algo_name> \
    --data-path ./output/smoke_<algo_name> \
    --save-path ./output/checkpoints/<algo_name>_smoke \
    --draft-vocab-size 32000 \
    --epochs 2 \
    --lr 1e-4 \
    --total-seq-len 2048 \
    --max-steps 100 \
    <ALGO_SPECIFIC_ARGS>
```

### Option B: Online training (with vLLM server)

Launch vLLM server on dedicated GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 /workspace/speculators/.venv/bin/python scripts/launch_vllm.py <VERIFIER_MODEL> \
    -- --port 8000 &
VLLM_PID=$!

# Wait for server
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done
```

Then train on remaining GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 /workspace/speculators/.venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    scripts/train.py \
    --verifier-name-or-path <VERIFIER_MODEL> \
    --speculator-type <algo_name> \
    --data-path ./output/smoke_<algo_name> \
    --vllm-endpoint http://localhost:8000/v1 \
    --save-path ./output/checkpoints/<algo_name>_smoke \
    --draft-vocab-size 32000 \
    --epochs 2 \
    --lr 1e-4 \
    --total-seq-len 2048 \
    --on-missing generate \
    --on-generate delete \
    --max-steps 100 \
    <ALGO_SPECIFIC_ARGS>
```

## Step 4: Monitor Training

Run the training command in the background and monitor output. Look for:

### Success indicators
- Loss values printed and decreasing over steps
- No NaN or Inf in loss
- No CUDA OOM errors
- No shape mismatch errors
- Accuracy metrics (if reported) improving
- Checkpoint saved message at end

### Failure indicators
- `RuntimeError: shape mismatch` — architecture bug in forward()
- `KeyError` or `AttributeError` — missing config field or model attribute
- `NaN` loss — learning rate too high or numerical instability
- `CUDA out of memory` — reduce batch size, seq length, or model size
- `NotImplementedError` — missing method implementation
- Registration errors — model type not found in registry

### What to do on failure
1. Read the full traceback
2. Identify the root cause (shape mismatch, missing field, etc.)
3. Fix the issue in the implementation files
4. Re-run `make quality` to ensure the fix is clean
5. Re-run training

## Step 5: Report Results

After training completes (or 100 steps), report:

```
## Smoke Training Results: <algo_name>

### Configuration
- Verifier: <model>
- GPUs: <count> x <type>
- Steps: <N>
- Samples: 500
- Sequence length: 2048

### Results
- Initial loss: X.XXX
- Final loss: X.XXX
- Loss trend: decreasing/stable/diverging
- Training time: X minutes
- Checkpoint saved: yes/no

### Verdict
- PASS: Loss decreased, no errors, checkpoint saved
- FAIL: [describe what went wrong]

### Next Steps
- [If PASS] Ready for full training run. Suggest hyperparameters for production training.
- [If FAIL] [Describe fixes needed]
```

Kill any background vLLM server:
```bash
kill $VLLM_PID 2>/dev/null || true
```
