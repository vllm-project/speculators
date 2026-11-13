# Quantize a drafter with llm-compressor

These examples quantize speculator drafters with llm-compressor. The worked example uses `RedHatAI/Qwen3.8-27B-speculator.dspark` as the drafter and `Qwen/Qwen3.8-27B` as its verifier; adapt these IDs and the model-specific settings below to your checkpoint. FP8 is data-free. The NVFP4 example is DSpark-specific and uses GPTQ with iMatrix, on-policy responses, and verifier hidden states for calibration.

## 1. Environment setup

Run commands from the repository root. Create separate Python 3.12 environments for the vLLM servers and for the data-generation clients and quantization scripts. We recommend `uv` for virtual-environment management. Install a hardware-appropriate PyTorch build if needed.

```bash
uv venv \
  --python 3.12 \
  vllm_venv
uv pip install \
  --python vllm_venv/bin/python \
  'vllm==0.33.0'
uv venv \
  --python 3.12 \
  .venv-quant
uv pip install \
  --python .venv-quant/bin/python \
  -r examples/quantization/requirements.txt
```

The quantization environment installs Speculators and its matching `hs-connectors` dependency through `requirements.txt`. The file-based hidden-state extraction path uses vLLM's built-in connector, so the vLLM environment does not require Speculators or `hs-connectors`. The quantization environment pins llm-compressor because the NVFP4 example uses its 0.15.0 interfaces.

Set these variables in each terminal. Use an unquantized drafter and a fresh output directory for each quantization run.

```bash
MODEL=Qwen/Qwen3.8-27B
DRAFT=RedHatAI/Qwen3.8-27B-speculator.dspark
RUN="$PWD/local/dspark-quantization"
mkdir -p "$RUN"
```

Hub IDs load their default revision. To pin a drafter revision, download it separately and set `DRAFT` to the local directory. Record the verifier and drafter revisions used.

## 2. Data-free FP8

The [FP8 example](fp8.py) uses llm-compressor's `model_free_ptq`, which requires safetensors weights and quantizes the checkpoint without calibration data or a model forward. Both schemes use static weight scales and dynamic activation scales:

- `FP8_DYNAMIC` (the default): per-channel weight scales and per-token activation scales.
- `FP8_BLOCK`: 128×128-block weight scales and token-group activation scales.

```bash
source .venv-quant/bin/activate
python examples/quantization/fp8.py \
  --model "$DRAFT" \
  --scheme FP8_DYNAMIC \
  --output-dir "$RUN/draft-fp8"
```

Pass `--scheme FP8_BLOCK` for blockwise FP8, with a fresh output directory. Check that your hardware and serving runtime support the selected scheme.

Before adapting the script, compare its top-level `IGNORE` module-name patterns with your checkpoint's module names. The default recipe excludes token embeddings, the LM and verifier output heads, and DSpark's Markov and confidence heads. Keep embeddings, output heads, and other special heads unquantized as appropriate for your architecture; the DSpark exclusions are not a universal recipe.

## 3. Calibrated NVFP4 for DSpark

The [NVFP4 example](dspark_nvfp4_gptq_imatrix.py) requires prepared on-policy responses and matching verifier hidden states. Set the calibration variables in each terminal that needs them:

```bash
SAMPLES=1024
SEQ_LENGTH=2048
BLOCK_SIZE=8
REGEN_LIMIT=1280
TARGET_LAYER_IDS=(4 12 20 28 36 44 52 60)
MIN_VALID_TOKENS=$((BLOCK_SIZE + 1))
```

For the DSpark example, `BLOCK_SIZE` is the drafter's block size and `TARGET_LAYER_IDS` must match the drafter's `aux_hidden_state_layer_ids` in order. `MIN_VALID_TOKENS` is `BLOCK_SIZE + 1` so each row has an anchor before its final draft block. Use the same `SEQ_LENGTH` for data preparation and NVFP4 calibration, and set the vLLM server's maximum model length to at least `SEQ_LENGTH + 1`. `REGEN_LIMIT` can exceed `SAMPLES` because preparation filters out ineligible rows.

The following resource figures are examples for this checkpoint and these settings, not general requirements. Full-length 2048-token BF16 caches can require approximately 180 GiB of disk for 1024 rows, plus staging and checkpoint space. During quantization, llm-compressor's basic pipeline caches calibration inputs in host RAM: eight layers of width 5120 at 1024 × 2048 tokens require approximately 160 GiB for BF16 hidden states alone, plus other inputs and overhead. Shorter rows need less.

### Regenerate and prepare responses

See the [response regeneration guide](../../docs/user_guide/tutorials/response_regeneration.md) for the full workflow and dataset options, including [local JSON/JSONL prompt files](../../docs/cli/response_regeneration.md#local-files). If you already have on-policy responses, skip regeneration and start with `speculators prepare-data`; see [Prepare Your Data](../../docs/user_guide/tutorials/train.md#step-1-prepare-your-data). The example below uses the `magpie` preset; choose data representative of your deployment.

In a server terminal, activate the vLLM environment and start the verifier:

```bash
source vllm_venv/bin/activate
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len "$((SEQ_LENGTH + 1))"
```

After `curl -fsS http://127.0.0.1:8000/v1/models` succeeds, regenerate responses in a second terminal, then prepare them:

```bash
source .venv-quant/bin/activate
speculators regenerate-responses \
  --dataset magpie \
  --model "$MODEL" \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit "$REGEN_LIMIT" \
  --concurrency 4 \
  --max-tokens 1024 \
  --sampling-params '{"temperature":0.6,"top_p":0.95,"seed":42}' \
  --outfile "$RUN/responses.jsonl" \
  --resume
speculators prepare-data \
  --model "$MODEL" \
  --data "$RUN/responses.jsonl" \
  --output "$RUN/prepared" \
  --max-samples "$SAMPLES" \
  --seq-length "$SEQ_LENGTH" \
  --minimum-valid-tokens "$MIN_VALID_TOKENS" \
  --seed 42
python - "$RUN/prepared" <<'PY'
import sys
from datasets import load_from_disk

print(len(load_from_disk(sys.argv[1])))
PY
```

Confirm that the printed count is at least `$SAMPLES`. If filtering leaves fewer rows, increase `REGEN_LIMIT` and prepare a fresh run.

### Extract hidden states offline

See [Get Hidden States](../../docs/user_guide/tutorials/train.md#step-3-get-hidden-states) for the offline extraction workflow. Stop the regeneration server, then start an extraction server on a filesystem shared with the client. The launcher appends the final verifier layer to the requested draft layers by default.

```bash
python scripts/launch_vllm.py train "$MODEL" \
  --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
  --hidden-states-path "$RUN/staging" \
  --provenance-dir "$RUN" \
  -- \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len "$((SEQ_LENGTH + 1))"
```

After `curl -fsS http://127.0.0.1:8001/v1/models` succeeds, run in the quantization terminal:

```bash
speculators generate-offline-data \
  --model "$MODEL" \
  --endpoint http://127.0.0.1:8001/v1 \
  --preprocessed-data "$RUN/prepared" \
  --output "$RUN/hidden_states" \
  --max-samples "$SAMPLES" \
  --concurrency 4 \
  --validate-outputs \
  --fail-on-error
```

Stop the server before quantization. Use a fresh cache for each set of rows and extraction settings. The adapter checks token alignment, shape, layer count, and finite values; it accepts FP16, BF16, and FP32 states, not raw scaled FP8. The cache does not record layer IDs, so keep the launcher provenance with the cache and preserve the drafter's `aux_hidden_state_layer_ids` order, optionally followed by the final verifier layer.

### Quantize with NVFP4

```bash
python examples/quantization/dspark_nvfp4_gptq_imatrix.py \
  --model "$DRAFT" \
  --data-dir "$RUN/prepared" \
  --hidden-states-dir "$RUN/hidden_states" \
  --samples "$SAMPLES" \
  --sequence-length "$SEQ_LENGTH" \
  --output-dir "$RUN/draft-nvfp4"
```

DSpark's training forward computes verifier targets, logits, and losses, but calibration needs only backbone activations and the compressor may disable the output head. The example supplies a [loss-free calibration forward](dspark_calibration.py) that runs the same backbone. It also replaces llm-compressor 0.15.0's model validator because Transformers `AutoConfig` cannot read the speculators config; the replacement accepts only unquantized DSpark checkpoints.

NVFP4 calibrates the whole model. The drafter loads verifier weights by the name in its config; ensure that this resolves to the same verifier revision used for generation and extraction. Record revisions, package and CUDA versions, prompt source and checksum, commands, seed, selected rows, and recipe alongside the outputs. Keep the launcher provenance and patches with the run.

NVFP4 serving depends on hardware and runtime support. Do not assume native W4A4 execution or quality recovery; validate both on your target stack.

## 4. Deploy and validate

In the vLLM terminal, set `QUANTIZED_DRAFT` and a fresh `EVAL_RUN` to matching FP8 or NVFP4 paths:

```bash
QUANTIZED_DRAFT="$RUN/draft-fp8"
EVAL_RUN="$RUN/eval-fp8"
# For NVFP4, instead use:
# QUANTIZED_DRAFT="$RUN/draft-nvfp4"
# EVAL_RUN="$RUN/eval-nvfp4"
```

Serve the selected output with the matching verifier. `--spec-method dspark` and `--spec-tokens 7` are DSpark example settings; adapt them to your drafter.

```bash
python scripts/launch_vllm.py eval "$MODEL" \
  --spec-model "$QUANTIZED_DRAFT" \
  --spec-tokens 7 \
  --spec-method dspark \
  --provenance-dir "$EVAL_RUN" \
  -- \
  --port 9145 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --per-request-spec-decode-metrics summary \
  --no-enable-prefix-caching
```

Wait for `curl -fsS http://127.0.0.1:9145/v1/models` to succeed, then send a request from another terminal:

```bash
curl -fsS http://127.0.0.1:9145/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Explain photosynthesis briefly.\",
    \"max_tokens\": 64,
    \"temperature\": 0
  }"
```

Check for nonempty output and accepted draft tokens in `metrics.speculative_decoding`. Evaluate each quantized drafter on a representative dataset before drawing quality conclusions.
