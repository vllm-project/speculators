# Diagnostic: the speculator server with pinned scheduler limits

Why the speculator configuration queues at N = 128 (median TTFT 11 s) while the plain server does not. Measured on 2026-10-06 with the same server as the main results plus `--max-num-seqs 256 --max-num-batched-tokens 16384`, HumanEval, N = 96 and 128, 60 s warmup and 120 s window, while vLLM's `num_requests_running`, `num_requests_waiting` and `kv_cache_usage_perc` gauges were sampled every 2 s (`gauges.csv`).

| N   | tok/s | tok/s per user | median TTFT | max running | waiting | KV cache usage |
| --- | ----- | -------------- | ----------- | ----------- | ------- | -------------- |
| 96  | 3,888 | 57             | 4.6 s       | 74          | 24      | 99.9%          |
| 128 | 3,897 | 57             | 10.9 s      | 74          | 56      | 99.9%          |

The batch cap was 256, so the cap is not the limit. The KV cache is: it holds 544k token-equivalents with the drafter loaded (15 cache groups) against 1.36M without it (4 groups), and the hybrid Qwen3.8 architecture reserves per-request state, so the server runs out of cache at about 73 concurrent requests and admits the rest only as others finish. The N = 128 point therefore measures queueing plus decode, which is what `validate` check 8 reports. Both points reproduce the main results within 2%, so pinning the limits changes nothing below the cache limit. A quantized target such as `RedHatAI/Qwen3.8-27B-NVFP4` frees memory for the cache; the model card suggests it.
