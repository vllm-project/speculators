CONDA_ENV="speculators_v3"


FIXED_ACCEPTANCE_RATE=-1 \
CUDA_VISIBLE_DEVICES=7 \
VLLM_USE_V1=1 \
vllm serve /home/linghao/models/Qwen3-30B-A3B/ \
  --seed 42 \
  -tp 1 \
  --max_model_len 1024 \
  --enable_chunked_prefill \
  --speculative-config '{
    "model": "/nm/drive0/linghao/Qwen3-30B-A3B-speculator",
    "num_speculative_tokens": 10,
    "method": "eagle",
    "draft_tensor_parallel_size": 1
  }' \
  --port 8007


GUIDELLM__MAX_CONCURRENCY=256 \
GUIDELLM__PREFERRED_ROUTE="chat_completions" \
guidellm benchmark \
  --target "http://localhost:8002/v1" \
  --model "/home/linghao/models/Qwen3-30B-A3B/" \
  --output-path ~/speculators/throughput/output_Qwen3-30B-A3B_RS_draft-3.json \
  --data "prompt_tokens=512,output_tokens=128" \
  --rate-type sweep \
  --max-seconds 90 \
  --backend-args '{
    "extra_body": {
      "chat_completions": {
        "temperature": 0.6
      },
      "max_output_tokens":1024
    }
  }'