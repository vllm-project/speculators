# Evaluating Model Performance

This tutorial shows you how to benchmark and evaluate your trained speculator models using GuideLLM and vLLM metrics.

## Overview

After training a speculator, it's crucial to measure its performance to understand:

- **Speedup:** How much faster is inference compared to the base model?
- **Acceptance rate:** How many draft tokens are accepted on average?
- **Throughput:** Requests per second with vs without speculation
- **Latency:** Time to first token and time per output token

**What you'll learn:**

- How to set up the evaluation framework
- How to benchmark with GuideLLM
- How to interpret acceptance metrics
- How to compare against base model performance

**Time required:** ~30-60 minutes

**Prerequisites:**

- Trained speculator model (or use pre-trained from HuggingFace)
- vLLM 0.12.1 or greater
- GuideLLM installed

## Step 1: Install Dependencies

```bash
cd examples/evaluate/eval-guidellm

# Quick install
bash setup.sh

# Or with uv for faster installation
bash setup.sh --use-uv
```

This installs:

- GuideLLM (benchmarking framework)
- Required Python packages
- Evaluation scripts

## Step 2: Quick Start - Pre-configured Models

Test with a pre-trained model using provided configs:

```bash
# Llama-3.1-8B EAGLE3
./run_evaluation.sh -c configs/llama-3.1-8b-eagle3.env

# Qwen3-8B EAGLE3
./run_evaluation.sh -c configs/qwen3-8b-eagle3.env

# Llama-3.3-70B EAGLE3
./run_evaluation.sh -c configs/llama-3.3-70b-eagle3.env
```

The script will:

1. Start a vLLM server with speculative decoding
2. Run GuideLLM benchmarks
3. Extract and analyze metrics
4. Generate a report

**Output:**

```
eval_results_20260416_123456/
├── guidellm_results.html      # Interactive benchmark report
├── guidellm_results.json      # Raw benchmark data
├── vllm_server.log            # vLLM server logs
├── acceptance_metrics.txt     # Extracted metrics
└── summary.txt                # Performance summary
```

## Step 3: Evaluate Your Custom Model

### Basic Evaluation

```bash
./run_evaluation.sh \
  -b "meta-llama/Llama-3.1-8B-Instruct" \
  -s "./checkpoints/best" \
  -d "emulated"
```

**Parameters:**

- `-b` - Base/target model
- `-s` - Your trained speculator model
- `-d` - Dataset for benchmarking

### Advanced Evaluation

```bash
./run_evaluation.sh \
  -b "meta-llama/Llama-3.1-8B-Instruct" \
  -s "./checkpoints/best" \
  -d "RedHatAI/speculator_benchmarks:math_reasoning.jsonl" \
  -o ./my_evaluation
```

## Step 4: Understanding Datasets

### Available Datasets

**Emulated (synthetic):**

```bash
-d "emulated"  # Quick synthetic benchmark
```

- Generates random prompts
- Fast, reproducible
- Good for initial testing

**Real datasets:**

```bash
# Math reasoning
-d "RedHatAI/speculator_benchmarks:math_reasoning.jsonl"

# Code generation
-d "RedHatAI/speculator_benchmarks:code_generation.jsonl"

# Creative writing
-d "RedHatAI/speculator_benchmarks:creative_writing.jsonl"
```

### Custom Dataset

Use your own JSONL file:

```json
{"prompt": "What is machine learning?"}
{"prompt": "Explain quantum computing."}
{"prompt": "Write a Python function to sort a list."}
```

Then:

```bash
-d "./my_prompts.jsonl"
```

## Step 5: Reading the Results

### Acceptance Metrics

In `acceptance_metrics.txt`:

```
Average Acceptance Length: 2.3
Acceptance Rate by Position:
  Position 1: 65%
  Position 2: 45%
  Position 3: 28%
  Position 4: 15%
  Position 5: 8%

Total Draft Tokens: 125,000
Total Accepted Tokens: 287,500
Overall Acceptance: 2.3 tokens/speculation
```

**Key metrics:**

- **Average acceptance:** Higher is better (typical: 1.5-3.0)
- **Position 1 accuracy:** First token prediction rate
- **Acceptance rate curve:** Should decrease by position

### GuideLLM Report

Open `guidellm_results.html` in a browser to see:

- **Throughput comparison:** Requests/second with vs without speculator
- **Latency metrics:** TTFT (time to first token), TPOT (time per output token)
- **Token generation rate:** Tokens per second
- **Load testing results:** Performance under various request rates

### Performance Summary

In `summary.txt`:

```
=== Speculative Decoding Performance ===

Configuration:
- Base Model: meta-llama/Llama-3.1-8B-Instruct
- Speculator: ./checkpoints/best
- Speculative Tokens: 5

Results:
- Speedup: 2.1x
- Throughput: 45.3 req/s (vs 21.6 req/s baseline)
- Average Acceptance: 2.3 tokens
- Latency Reduction: 35%
```

## Step 6: Comparing Performance

### Baseline vs Speculative

Run both for comparison:

```bash
# Baseline (no speculator)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 &
guidellm --target http://localhost:8000/v1 --output baseline_results

# Speculative decoding
vllm serve ./checkpoints/best --port 8001 &
guidellm --target http://localhost:8001/v1 --output speculative_results
```

Compare:

- Throughput improvement
- Latency reduction
- Quality (should be identical)

## Step 7: Advanced Benchmarking

### Load Testing

Test under various request rates:

```bash
# Low load (1 req/s)
guidellm --target http://localhost:8000/v1 --request-rate 1.0

# Medium load (10 req/s)
guidellm --target http://localhost:8000/v1 --request-rate 10.0

# High load (50 req/s)
guidellm --target http://localhost:8000/v1 --request-rate 50.0
```

**Expected behavior:**

- Low batch sizes: Higher speedup (2.5-3x)
- High batch sizes: Lower speedup (1.5-2x)

### Different Prompt Lengths

```bash
# Short prompts (512 tokens)
guidellm --data "prompt_tokens=512,generated_tokens=256"

# Medium prompts (1024 tokens)
guidellm --data "prompt_tokens=1024,generated_tokens=512"

# Long prompts (2048 tokens)
guidellm --data "prompt_tokens=2048,generated_tokens=1024"
```

**Expected behavior:**

- Longer prompts: Often higher speedup
- Varies by model and workload

## Step 8: Analyzing Acceptance Patterns

### Extract Detailed Metrics

```python
# scripts/parse_logs.py usage
python scripts/parse_logs.py vllm_server.log
```

Outputs:

```
Position-wise acceptance rates:
Pos 1: 67.2% (4,032 / 6,000)
Pos 2: 48.5% (2,910 / 6,000)
Pos 3: 31.2% (1,872 / 6,000)
Pos 4: 18.7% (1,122 / 6,000)
Pos 5: 9.3% (558 / 6,000)

Draft token distribution:
1 token:  12.3%
2 tokens: 23.5%
3 tokens: 31.2%
4 tokens: 19.8%
5 tokens: 13.2%

Average acceptance: 2.28 tokens
```

### Understanding Acceptance Curves

**Good performance:**

```
Pos 1: 70% → Pos 2: 50% → Pos 3: 30% → Pos 4: 15% → Pos 5: 5%
Average: 2.4 tokens
```

**Poor performance:**

```
Pos 1: 40% → Pos 2: 15% → Pos 3: 5% → Pos 4: 1% → Pos 5: 0%
Average: 0.8 tokens
```

**Indicators of issues:**

- Pos 1 < 50%: Model not well-aligned with target
- Steep dropoff: Try fewer speculative tokens
- Flat curve: Underfitting, train longer

## Performance Optimization

### If Acceptance Rate is Low

**1. Train longer:**

```bash
--epochs 20  # Instead of 10
```

**2. Use more training data:**

```bash
--max-samples 100000  # Instead of 10000
```

**3. Check data quality:** Ensure preprocessing was correct, data matches your use case.

**4. Adjust speculative tokens:**

```bash
# Fewer tokens may have higher acceptance
vllm serve model --speculative-tokens 3  # Instead of 5
```

### If Speedup is Lower Than Expected

**1. Reduce batch size:**

```bash
# Speculative decoding works best at low batch sizes
--max-num-seqs 8  # Instead of 256
```

**2. Check GPU utilization:**

```bash
nvidia-smi
```

If draft model is slow, may need optimization.

**3. Verify draft vocab size:** Smaller vocab = faster inference

```bash
# Check model config
cat checkpoints/best/config.json | grep draft_vocab_size
```

## Benchmarking Best Practices

1. **Warm up the server:** Run a few requests before measuring
2. **Use realistic prompts:** Match your actual use case
3. **Test multiple request rates:** Find optimal operating point
4. **Measure end-to-end:** Include network latency if relevant
5. **Compare apple-to-apples:** Same hardware, same settings
6. **Run multiple times:** Average over several runs for stability

## Creating Custom Evaluation Configs

```bash
# configs/my-model.env
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SPECULATOR_MODEL="./checkpoints/best"
NUM_SPEC_TOKENS=5
METHOD="eagle3"

# Dataset
DATASET="emulated"

# vLLM settings
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
PORT=8000

# Sampling
TEMPERATURE=0.7
TOP_P=0.9

# Output
OUTPUT_DIR="my_evaluation_$(date +%Y%m%d_%H%M%S)"
```

Run with:

```bash
./run_evaluation.sh -c configs/my-model.env
```

## Interpreting Results for Production

### Production Readiness Checklist

- [ ] Average acceptance > 1.5 tokens
- [ ] Position 1 accuracy > 50%
- [ ] Speedup > 1.5x at your target batch size
- [ ] Throughput improvement > 30%
- [ ] Quality verified (outputs identical to base model)
- [ ] Tested on representative workload
- [ ] Acceptable latency under load

### When to Iterate

**Good enough for production:**

- Avg acceptance: 2.0-2.5+
- Speedup: 2-3x
- Stable under load

**Needs improvement:**

- Avg acceptance: < 1.5
- Speedup: < 1.5x
- Unstable acceptance rates

**Action items if underperforming:**

1. Train with more data
2. Try different hyperparameters
3. Ensure data quality matches use case
4. Consider different algorithm (EAGLE-3 vs DFlash)

## Example: Complete Evaluation Workflow

```bash
#!/bin/bash

MODEL_NAME="llama31-8b-custom"
BASE="meta-llama/Llama-3.1-8B-Instruct"
SPEC="./checkpoints/best"

# 1. Quick synthetic test
echo "=== Quick Test ==="
./run_evaluation.sh -b $BASE -s $SPEC -d "emulated" -o results/${MODEL_NAME}_quick

# 2. Math reasoning dataset
echo "=== Math Reasoning ==="
./run_evaluation.sh \
  -b $BASE \
  -s $SPEC \
  -d "RedHatAI/speculator_benchmarks:math_reasoning.jsonl" \
  -o results/${MODEL_NAME}_math

# 3. Code generation
echo "=== Code Generation ==="
./run_evaluation.sh \
  -b $BASE \
  -s $SPEC \
  -d "RedHatAI/speculator_benchmarks:code_generation.jsonl" \
  -o results/${MODEL_NAME}_code

# 4. Custom workload
echo "=== Custom Workload ==="
./run_evaluation.sh \
  -b $BASE \
  -s $SPEC \
  -d "./my_production_prompts.jsonl" \
  -o results/${MODEL_NAME}_production

# 5. Aggregate results
echo "=== Results Summary ==="
echo "Quick Test:"
cat results/${MODEL_NAME}_quick/summary.txt
echo ""
echo "Math:"
cat results/${MODEL_NAME}_math/acceptance_metrics.txt
echo ""
echo "Code:"
cat results/${MODEL_NAME}_code/acceptance_metrics.txt
echo ""
echo "Production:"
cat results/${MODEL_NAME}_production/acceptance_metrics.txt
```

## Troubleshooting

### vLLM Server Won't Start

```bash
# Check logs
tail -f vllm_server.log

# Verify model exists
ls -la ./checkpoints/best/

# Check GPU availability
nvidia-smi
```

### Low Acceptance Rates

1. Check training converged properly
2. Verify same target model used for training and serving
3. Ensure hidden state layers match
4. Review training data quality

### GuideLLM Errors

```bash
# Update GuideLLM
pip install --upgrade guidellm

# Check server is responsive
curl http://localhost:8000/v1/models

# Verify dataset format
head my_dataset.jsonl
```

## Next Steps

After evaluating your model:

1. **Deploy to production** - [Serve in vLLM Tutorial](serve_vllm.md)
2. **Optimize further** - Iterate on training if needed
3. **A/B test** - Compare with base model in production
4. **Monitor in production** - Track acceptance rates live
5. **Share results** - Document and share with team

## Summary

You've learned how to:

- ✅ Set up the evaluation framework
- ✅ Benchmark with GuideLLM
- ✅ Interpret acceptance metrics and speedup
- ✅ Optimize based on results
- ✅ Create custom evaluation configs
- ✅ Validate production readiness

Proper evaluation ensures your speculator provides real value in production. Use these metrics to guide training iterations and deployment decisions.

## See Also

- [Train EAGLE-3 Online](train_eagle3_online.md) - Training tutorial
- [Train EAGLE-3 Offline](train_eagle3_offline.md) - Offline training
- [Serve in vLLM](serve_vllm.md) - Production deployment
- [EAGLE-3 Algorithm](../algorithms/eagle3.md) - Algorithm details
- [GuideLLM Documentation](https://github.com/vllm-project/guidellm) - Benchmarking tool docs
