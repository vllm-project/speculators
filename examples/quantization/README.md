# Quantize a DSpark drafter

These examples quantize `RedHatAI/Qwen3.8-27B-speculator.dspark` for the verifier `Qwen/Qwen3.8-27B`. FP8 is data-free; NVFP4 uses GPTQ with iMatrix and verifier hidden states for calibration. Both leave embeddings, vocabulary, Markov and confidence heads unquantized.

## 1. Environment setup

Run commands from the repository root. Create separate Python 3.12 environments for the vLLM servers and for the data-generation clients and quantization scripts. Install a hardware-appropriate PyTorch build if needed.

```bash
uv venv --python 3.12 vllm_venv
uv pip install --python vllm_venv/bin/python 'vllm==0.33.0' 'speculators==0.9.0'
uv venv --python 3.12 .venv-quant
uv pip install --python .venv-quant/bin/python -r examples/quantization/requirements.txt
```

Speculators installs its matching `hs-connectors` dependency. The quantization environment also pins llm-compressor because the NVFP4 example uses its 0.15.0 interfaces.

Set these variables in each terminal. Both scripts require an unquantized drafter and a fresh output directory.

```bash
MODEL=Qwen/Qwen3.8-27B
DRAFT=RedHatAI/Qwen3.8-27B-speculator.dspark
RUN="$PWD/local/dspark-calibration-1024"
SAMPLES=1024
mkdir -p "$RUN"
```

Hub IDs load their default revision. To pin a drafter revision, download it separately and set `DRAFT` to the local directory. Record the verifier and drafter revisions used.

## 2. Generate calibration data

Skip this step for data-free FP8. For NVFP4, regenerate responses to representative prompts, prepare the tokenized rows, then extract verifier hidden states. The example uses the `magpie` preset; choose data representative of your deployment. See the [response regeneration guide](../../docs/user_guide/tutorials/response_regeneration.md) for other datasets and the [offline data generation guide](../../docs/user_guide/tutorials/train.md#step-3-get-hidden-states) for extraction details.

Full-length 2048-token BF16 caches can require approximately 180 GiB of disk for 1024 rows, plus staging and checkpoint space. During quantization, llm-compressor's basic pipeline caches calibration inputs in host RAM: eight layers of width 5120 at 1024 × 2048 tokens require approximately 160 GiB for BF16 hidden states alone, plus other inputs and overhead. Shorter rows need less.

### Regenerate and prepare responses

In a server terminal, activate the vLLM environment and start the verifier. Record the command, model revision, versions and sampling parameters in `$RUN`.

```bash
source vllm_venv/bin/activate
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8000/v1/models` succeeds, use a second terminal for the client:

```bash
source .venv-quant/bin/activate
speculators regenerate-responses --dataset magpie \
  --model "$MODEL" --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit 1280 --concurrency 4 --max-tokens 1024 \
  --sampling-params '{"temperature":0.6,"top_p":0.95,"seed":42}' \
  --outfile "$RUN/responses.jsonl" --resume
speculators prepare-data --model "$MODEL" --data "$RUN/responses.jsonl" \
  --output "$RUN/prepared" --max-samples "$SAMPLES" --seq-length 2048 \
  --minimum-valid-tokens 9 --seed 42
python -c 'import sys; from datasets import load_from_disk; print(len(load_from_disk(sys.argv[1])))' "$RUN/prepared"
```

The minimum of nine supervised tokens ensures an anchor before the final eight-token block. Confirm the printed count is at least `$SAMPLES`; fewer rows may be eligible than the regeneration limit. If needed, increase `--limit` and prepare a fresh run.

### Extract hidden states offline

Stop the generation server. In the vLLM terminal, start an extraction server on a filesystem shared with the client. The launcher appends verifier layer 64 to the eight draft layers.

```bash
python scripts/launch_vllm.py train "$MODEL" \
  --target-layer-ids 4 12 20 28 36 44 52 60 \
  --hidden-states-path "$RUN/staging" --provenance-dir "$RUN" \
  -- --host 127.0.0.1 --port 8001 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8001/v1/models` succeeds, run in the quantization terminal:

```bash
speculators generate-offline-data --model "$MODEL" \
  --endpoint http://127.0.0.1:8001/v1 --preprocessed-data "$RUN/prepared" \
  --output "$RUN/hidden_states" --max-samples "$SAMPLES" --concurrency 4 \
  --validate-outputs --fail-on-error
```

Stop the server before quantization. Use a fresh cache for each set of rows and extraction settings. The adapter checks token alignment, shape, layer count and finite values; it accepts FP16, BF16 and FP32 states, not raw scaled FP8. The cache does not record layer IDs: capture them in the drafter's `target_layer_ids` order, optionally followed by the final verifier layer, and keep the launcher provenance with the cache.

## 3. Quantize

Activate `.venv-quant` and choose an example. Data-free schemes do not need calibration inputs: llm-compressor's `model_free_ptq` quantizes the checkpoint without running the model. The [FP8 example](dspark_fp8.py) defaults to `FP8_DYNAMIC`; its recipe shows how to select `FP8_BLOCK`.

```bash
source .venv-quant/bin/activate
python examples/quantization/dspark_fp8.py --model "$DRAFT" \
  --output-dir "$RUN/draft-fp8"
```

The [NVFP4 example](dspark_nvfp4_gptq_imatrix.py) uses GPTQ and the iMatrix observer with the prepared rows and hidden states from step 2. DSpark's training forward computes verifier targets, logits and losses, while calibration only needs backbone activations and the compressor may disable the output head. The example supplies a [loss-free calibration forward](dspark_calibration.py) that runs the same backbone. It also replaces llm-compressor 0.15.0's model validator because Transformers `AutoConfig` cannot read the speculators config; the replacement accepts only unquantized DSpark checkpoints.

```bash
python examples/quantization/dspark_nvfp4_gptq_imatrix.py --model "$DRAFT" \
  --data-dir "$RUN/prepared" --hidden-states-dir "$RUN/hidden_states" \
  --samples "$SAMPLES" --output-dir "$RUN/draft-nvfp4"
```

NVFP4 calibrates the whole model. The drafter loads verifier weights by the name in its config; ensure it resolves to the same verifier revision used for generation and extraction. Record revisions, package and CUDA versions, prompt source and checksum, commands, seed, selected rows and recipe alongside the outputs. Keep the launcher provenance and patches with the run.

## 4. Deploy and validate

In the vLLM terminal, serve each output separately with the matching verifier and a fresh provenance directory:

```bash
python scripts/launch_vllm.py eval "$MODEL" --spec-model "$RUN/draft-fp8" \
  --spec-tokens 7 --spec-method dspark --provenance-dir "$RUN/eval-fp8" \
  -- --port 9145 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --enforce-eager --per-request-spec-decode-metrics summary --no-enable-prefix-caching
```

Wait for `curl -fsS http://127.0.0.1:9145/v1/models` to succeed, then send a request from another terminal:

```bash
curl -fsS http://127.0.0.1:9145/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B","prompt":"Explain photosynthesis briefly.","max_tokens":64,"temperature":0}'
```

Check for nonempty output and accepted draft tokens in `metrics.speculative_decoding`, then repeat with `draft-nvfp4` and `eval-nvfp4`. Evaluate on a representative dataset before drawing quality conclusions.

On H100, llm-compressor 0.15.0 completed both scripts on the target checkpoint; NVFP4 used 128 offline calibration rows. Both outputs reloaded and served a speculative completion with vLLM 0.33.0. NVFP4 serving warned that fused layers have different input global scales, which may reduce accuracy. The H100 Humming path uses BF16 activations; native NVFP4 W4A4 execution and recovery quality were not measured.
