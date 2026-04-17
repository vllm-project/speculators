# Serve in vLLM

This tutorial shows you how to deploy your trained speculator model for production inference using vLLM.

## Overview

After training a speculator, deploying it with vLLM is straightforward. vLLM automatically recognizes the `speculators_config` in your model and enables speculative decoding.

**What you'll learn:**
- How to serve speculator models with vLLM
- How to customize speculative decoding parameters
- How to optimize for production
- How to monitor and debug serving

## Quick Start

### Basic Serving

```bash
vllm serve ./checkpoints/best
```

That's it! vLLM will:
1. Load your speculator model
2. Read the `speculators_config` from config.json
3. Load the target/verifier model
4. Enable speculative decoding automatically

### Verify It's Working

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="./checkpoints/best",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

Check vLLM logs for speculative decoding metrics:
```
INFO: Speculative decoding enabled
INFO: Average acceptance: 2.3 tokens
```

## Serving Options

### Specify Port

```bash
vllm serve ./checkpoints/best --port 8080
```

### Multi-GPU (Tensor Parallelism)

```bash
vllm serve ./checkpoints/best \
  --tensor-parallel-size 4
```

### Custom Speculative Tokens

Override the default from config:

```bash
vllm serve ./checkpoints/best \
  --speculative-tokens 7
```

### GPU Memory Optimization

```bash
vllm serve ./checkpoints/best \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192
```

## Production Configuration

### High-Throughput Setup

```bash
vllm serve ./checkpoints/best \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --disable-log-requests
```

### Low-Latency Setup

```bash
vllm serve ./checkpoints/best \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-num-seqs 16 \      # Lower batch size
  --speculative-tokens 5
```

## Monitoring

### vLLM Metrics

vLLM logs show speculative decoding performance:

```
Speculative Stats:
- Draft tokens proposed: 50,000
- Tokens accepted: 115,000
- Average acceptance: 2.3 tokens
- Acceptance by position: [0.68, 0.52, 0.35, 0.18, 0.08]
```

### Prometheus Metrics

Enable Prometheus endpoint:

```bash
vllm serve ./checkpoints/best --enable-metrics
```

Access metrics at: `http://localhost:8000/metrics`

Key metrics:
- `vllm:spec_decode_draft_acceptance_rate`
- `vllm:spec_decode_efficiency`
- `vllm:request_latency_seconds`

## Customizing Speculative Decoding

### Via Config (Recommended)

Edit `config.json` in your model:

```json
{
  "speculators_config": {
    "proposal_methods": [
      {
        "proposal_type": "greedy",
        "speculative_tokens": 7,
        "verifier_accept_k": 1
      }
    ]
  }
}
```

### Via CLI Arguments

```bash
vllm serve ./checkpoints/best \
  --speculative-tokens 7 \
  --speculative-max-model-len 8192
```

## Deployment Patterns

### Docker Deployment

```dockerfile
FROM vllm/vllm-openai:latest

COPY ./checkpoints/best /model

CMD ["vllm", "serve", "/model", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t my-speculator .
docker run --gpus all -p 8000:8000 my-speculator
```

### Kubernetes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-speculator
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    args:
      - "vllm"
      - "serve"
      - "/model"
    resources:
      limits:
        nvidia.com/gpu: 4
    volumeMounts:
    - name: model
      mountPath: /model
```

### Load Balancing

Use multiple vLLM instances behind a load balancer:

```bash
# Instance 1
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./model --port 8000

# Instance 2
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ./model --port 8001
```

## Troubleshooting

### Model Not Loading

**Error:** `Model not found`

```bash
# Verify model path
ls -la ./checkpoints/best/config.json

# Check vLLM can access it
vllm serve ./checkpoints/best --trust-remote-code
```

### Speculative Decoding Not Enabled

**Check logs for:**
```
WARNING: Speculative decoding disabled
```

**Solutions:**
1. Verify `speculators_config` exists in config.json
2. Check verifier model is accessible
3. Ensure vocab mappings (d2t.pt, t2d.pt) are present

### Low Acceptance Rates in Production

**Monitor:**
```
Average acceptance: 0.8 tokens  # Too low!
```

**Possible causes:**
1. Different data distribution than training
2. Model not properly loaded
3. Incorrect target layers

**Solutions:**
1. Retrain on production-like data
2. Verify model integrity
3. Check configuration matches training

## Performance Tips

1. **Use tensor parallelism for large models:**
   ```bash
   --tensor-parallel-size 8
   ```

2. **Tune batch size for your workload:**
   ```bash
   --max-num-seqs 64  # Adjust based on latency/throughput trade-off
   ```

3. **Optimize speculative tokens:**
   ```bash
   --speculative-tokens 5  # Sweet spot often 3-7
   ```

4. **Enable KV cache optimization:**
   ```bash
   --enable-prefix-caching
   ```

5. **Use appropriate dtype:**
   ```bash
   --dtype bfloat16  # Or float16
   ```

## Summary

You've learned how to:
- ✅ Serve speculator models with vLLM
- ✅ Configure for production
- ✅ Monitor performance
- ✅ Deploy with Docker/Kubernetes
- ✅ Troubleshoot common issues

Speculator models integrate seamlessly with vLLM, providing lossless 2-3x speedup in production.

## See Also

- [Evaluating Performance](evaluating_performance.md) - Benchmark before deploying
- [Training Feature](../features/training.md) - Training documentation
- [vLLM Documentation](https://docs.vllm.ai/) - Complete vLLM guide
