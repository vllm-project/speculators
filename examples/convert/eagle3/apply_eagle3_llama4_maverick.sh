speculators convert nvidia/Llama-4-Maverick-17B-128E-Eagle3 \
  --algorithm eagle3 \
  --verifier RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16 \
  --output-path Llama4-Maverick-Eagle3-Speculators \
  --validate-device cuda:0 \
  --algorithm-kwargs '{"eagle_aux_hidden_state_layer_ids": [1,23,44], "norm_before_residual": false}'
