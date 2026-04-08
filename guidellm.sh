guidellm benchmark \
  --target http://localhost:8000 \
  --data "RedHatAI/speculator_benchmarks" \
  --data-args '{"data_files": "HumanEval.jsonl"}' \
  --data-column-mapper '{"text_column":"prompt"}' \
  --profile throughput \
  --rate 16 \
  --max-seconds 180
