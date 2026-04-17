# Algorithm Decision Guide

This guide helps you choose the right speculative decoding algorithm for your use case. Currently, Speculators supports two algorithms: **EAGLE-3** and **DFlash**.

## Quick Recommendation

**For most users: Start with EAGLE-3**

EAGLE-3 is production-ready, broadly supported, and provides consistent performance across different workloads and model architectures.

## Decision Tree

```
┌─────────────────────────────────────────┐
│  Which model are you using?             │
└───────────┬─────────────────────────────┘
            │
            ├─── Llama, gpt-oss, MoE, VLM ──────► Use EAGLE-3 ✓
            │                                      (Only option)
            │
            └─── Qwen3 ────┐
                           │
                           ▼
            ┌──────────────────────────────────────────┐
            │  What type of content are you generating? │
            └───────────┬──────────────────────────────┘
                        │
                        ├─── Free-form conversation ──────► Use EAGLE-3 ✓
                        │    Creative writing                (Better performance)
                        │    General chat
                        │
                        ├─── Structured output ────┐
                        │    JSON, code, tables    │
                        │                          ▼
                        │              ┌────────────────────────────┐
                        │              │  Is this production-critical? │
                        │              └───────────┬────────────────┘
                        │                          │
                        │                          ├─── Yes ──────► Use EAGLE-3 ✓
                        │                          │                 (More stable)
                        │                          │
                        │                          └─── No ───────► Try DFlash
                        │                                            (Experiment)
                        │
                        └─── Unsure ──────────────────────► Use EAGLE-3 ✓
                                                            (Safe default)
```

## Detailed Comparison

### Model Support

| Model Family | EAGLE-3 | DFlash |
|--------------|---------|--------|
| Llama (all sizes) | ✅ | ❌ |
| Qwen3 (8B, 14B, 32B) | ✅ | ✅ |
| Qwen3 MoE (30B, 235B) | ✅ | ✅ |
| Qwen3-VL | ✅ | ⏳ |
| gpt-oss | ✅ | ❌ |
| Other architectures | ✅ | ❌ |

**Verdict:** EAGLE-3 supports all models, DFlash is Qwen3-only

### Performance by Workload

| Workload Type | EAGLE-3 | DFlash | Winner |
|---------------|---------|--------|--------|
| General conversation | 2.0-2.5x | 1.5-2.0x | EAGLE-3 |
| Creative writing | 2.2-2.8x | 1.5-2.2x | EAGLE-3 |
| Question answering | 2.0-2.5x | 1.8-2.3x | EAGLE-3 |
| JSON generation | 1.8-2.2x | 2.5-3.5x | DFlash |
| Code generation | 2.0-2.5x | 2.2-3.0x | DFlash (structured) |
| Table generation | 1.8-2.2x | 2.3-3.2x | DFlash |
| Mixed content | 2.0-2.5x | 1.6-2.5x | EAGLE-3 (consistent) |

**Verdict:**
- EAGLE-3 wins for general content
- DFlash can excel for structured/formatted output

### Production Readiness

| Aspect | EAGLE-3 | DFlash |
|--------|---------|--------|
| Maturity | Production-ready ✅ | Experimental ⏳ |
| vLLM integration | Fully optimized ✅ | Available, less optimized ⏳ |
| Real-world testing | Extensively tested ✅ | Limited testing ⚠️ |
| Documentation | Complete ✅ | Good ✅ |
| Community usage | Wide adoption ✅ | Early adopters ⏳ |
| Support | Full support ✅ | Community support ⏳ |

**Verdict:** EAGLE-3 is more suitable for production deployments

### Resource Requirements

| Resource | EAGLE-3 | DFlash |
|----------|---------|--------|
| Training time | Baseline | +0-20% (similar) |
| Training memory | Baseline | +10-15% (anchors) |
| Inference memory | Lower ✅ | Higher (anchor overhead) |
| Disk space (hidden states) | Baseline | Similar |
| Draft vocab size | 32K (smaller) ✅ | 64K (larger) |

**Verdict:** EAGLE-3 has lower resource requirements

### Ease of Use

| Aspect | EAGLE-3 | DFlash |
|--------|---------|--------|
| Setup complexity | Simple ✅ | Moderate |
| Hyperparameters | Fewer, well-established ✅ | More tuning needed |
| Training stability | Very stable ✅ | Stable |
| Debugging | Easier (more docs) ✅ | More complex |
| Performance tuning | Straightforward ✅ | Requires experimentation |

**Verdict:** EAGLE-3 is easier to use and tune

## Use Case Examples

### ✅ Use EAGLE-3 For

**1. General Chatbots**
```
User: Tell me about machine learning
Assistant: Machine learning is a subset of artificial intelligence...
```
- Unpredictable conversation flow
- Variable response lengths
- EAGLE-3 provides consistent 2-2.5x speedup

**2. Content Generation**
```
Write a blog post about climate change...
```
- Creative writing
- Long-form content
- Natural language flow

**3. Question Answering**
```
Q: What is the capital of France?
A: The capital of France is Paris...
```
- Short to medium responses
- General knowledge tasks

**4. Multi-Model Deployments**
```
Llama-3.1-70B, Qwen3-72B, gpt-oss-120B
```
- Need one algorithm for multiple models
- EAGLE-3 works across all architectures

**5. Production Services**
```
Customer support, virtual assistants, etc.
```
- Stability and reliability critical
- Proven performance needed

### 🔬 Try DFlash For (Qwen3 only)

**1. JSON API Generation**
```python
{
  "user_id": 12345,
  "name": "John Doe",
  "email": "john@example.com",
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}
```
- Structured format
- Block-aligned patterns
- DFlash can achieve 2.5-3.5x speedup

**2. Code Generation**
```python
def calculate_sum(numbers: list) -> int:
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total
```
- Function definitions
- Regular structure
- Can benefit from block-based prediction

**3. Table/CSV Generation**
```
Name,Age,City,Country
Alice,30,Paris,France
Bob,25,London,UK
Carol,35,Tokyo,Japan
```
- Regular row structure
- Fixed-width patterns

**4. Experimental Projects**
```
Research projects, benchmarking, etc.
```
- Exploring algorithm differences
- Willing to tune hyperparameters
- Qwen3 model available

## Decision Factors

### 1. Model Architecture

**If using Llama, gpt-oss, MoE (non-Qwen3), or VLM:**
→ Use EAGLE-3 (only option)

**If using Qwen3:**
→ Continue to next factors

### 2. Content Type

**Primarily free-form text (conversation, writing, Q&A):**
→ Use EAGLE-3

**Primarily structured output (JSON, code, tables):**
→ Consider DFlash (benchmark first)

**Mixed content:**
→ Use EAGLE-3 (more consistent)

### 3. Production Requirements

**Production-critical service:**
→ Use EAGLE-3 (proven stability)

**Research or experimentation:**
→ Can try DFlash

**User-facing application:**
→ Use EAGLE-3 (better support)

### 4. Performance Tolerance

**Need consistent, predictable performance:**
→ Use EAGLE-3

**Can tolerate variable performance:**
→ Can experiment with DFlash

### 5. Resource Constraints

**Limited GPU memory:**
→ Use EAGLE-3 (lower overhead)

**Limited disk space:**
→ Use EAGLE-3 (smaller vocab)

**Ample resources:**
→ Either algorithm works

## Migration Path

### Starting with EAGLE-3

If you start with EAGLE-3 and later want to try DFlash:

```bash
# 1. Train DFlash using same data
python scripts/train.py \
  --speculator-type dflash \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./same_data \
  --draft-vocab-size 64000 \
  --epochs 10

# 2. Benchmark both models
# 3. Compare on your specific workload
# 4. Choose based on results
```

### Starting with DFlash

If you find DFlash doesn't meet needs:

```bash
# Train EAGLE-3 as alternative
python scripts/train.py \
  --speculator-type eagle3 \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./same_data \
  --draft-vocab-size 32000 \
  --epochs 10
```

## Benchmarking Checklist

If unsure which algorithm to use, benchmark both:

- [ ] Train both EAGLE-3 and DFlash on your dataset
- [ ] Test on representative prompts from your workload
- [ ] Measure acceptance rate and speedup
- [ ] Test with realistic batch sizes
- [ ] Evaluate latency and throughput
- [ ] Consider ease of deployment and maintenance
- [ ] Make decision based on data

## Common Scenarios

### Scenario 1: Building a Chatbot

**Requirements:**
- Llama-3.1-70B base model
- General conversation
- Production deployment

**Recommendation:** EAGLE-3
- Only option for Llama models
- Proven for conversation
- Production-ready

### Scenario 2: Code Generation Service

**Requirements:**
- Qwen3-32B base model
- Structured code output
- Experimental project

**Recommendation:** Try both
1. Start with EAGLE-3 (safe default)
2. Benchmark DFlash (may excel for code)
3. Choose based on results

### Scenario 3: Multi-Modal Assistant

**Requirements:**
- Qwen3-VL model
- Mixed text and vision
- Production service

**Recommendation:** EAGLE-3
- Better VLM support
- More stable for production
- Consistent performance

### Scenario 4: JSON API Generator

**Requirements:**
- Qwen3-14B base model
- 100% JSON output
- Internal tool (not critical)

**Recommendation:** DFlash worth trying
- Structured output is DFlash's strength
- Can experiment safely
- Benchmark against EAGLE-3

### Scenario 5: Multi-Model Platform

**Requirements:**
- Multiple models (Llama, Qwen3, gpt-oss)
- Unified experience
- Production deployment

**Recommendation:** EAGLE-3
- Works across all models
- Simpler maintenance
- Consistent behavior

## Future Considerations

As the library evolves:

**DFlash Improvements:**
- Multi-architecture support planned
- vLLM optimizations in progress
- More production testing ongoing

**When to Reconsider:**
- DFlash gains support for your model architecture
- Your workload is primarily structured output
- DFlash performance improves in future releases

**Stay Updated:**
- Check [Roadmap](../../roadmap.md) for upcoming features
- Monitor [Release Notes] for algorithm improvements
- Join [Community](../../community.md) for discussions

## Summary Table

| Criterion | Choose EAGLE-3 | Choose DFlash |
|-----------|----------------|---------------|
| **Model** | Any supported model | Qwen3 only |
| **Content** | Free-form text | Structured output |
| **Status** | Production-ready | Experimental |
| **Support** | Broad, mature | Limited, growing |
| **Performance** | Consistent 2-2.5x | Variable 1.5-3.5x |
| **Resources** | Lower overhead | Higher overhead |
| **Use Case** | General purpose | Specialized |
| **Risk Tolerance** | Low (proven) | Higher (newer) |

## Final Recommendation

**Default Choice: EAGLE-3**

Choose EAGLE-3 if you:
- Want the safe, proven option
- Need broad model support
- Require production stability
- Have general-purpose workloads
- Are unsure which to choose

**Experimental Choice: DFlash**

Try DFlash if you:
- Use Qwen3 models exclusively
- Generate primarily structured output
- Can benchmark and experiment
- Are willing to tune hyperparameters
- Don't need production guarantees yet

**When in Doubt:**
Train both algorithms and benchmark on your actual workload. The best choice depends on your specific requirements and data.

## See Also

- [EAGLE-3 Algorithm](eagle3.md) - Detailed EAGLE-3 documentation
- [DFlash Algorithm](dflash.md) - Detailed DFlash documentation
- [Getting Started](../getting_started.md) - Training your first model
- [Evaluating Performance](../tutorials/evaluating_performance.md) - Benchmarking guide
