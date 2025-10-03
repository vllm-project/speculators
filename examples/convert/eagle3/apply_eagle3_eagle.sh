speculators convert nm-testing/Speculator-Qwen3-8B-Eagle3\
  --algorithm eagle3 \
  --verifier Qwen/Qwen3-8B \
  --output-path Speculator-Qwen3-8B-Eagle3-converted \
  --validate-device cuda:0
  --algorithm-kwargs '{"norm_before_residual": true}'