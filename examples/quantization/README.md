# Quantize a DSpark drafter

These examples target `RedHatAI/Qwen3.8-27B-speculator.dspark` with verifier `Qwen/Qwen3.8-27B`. FP8 uses model-free quantization; NVFP4 uses GPTQ with iMatrix calibration on verifier hidden states. Both leave embeddings, vocabulary, Markov and confidence heads unquantized.

From the repository root, create separate Python 3.12 environments for vLLM and quantization. The serving commands below use vLLM 0.33.0. Install a hardware-appropriate PyTorch build if needed:

```bash
uv venv --python 3.12 vllm_venv
uv pip install --python vllm_venv/bin/python 'vllm==0.33.0' ./hs_connectors
uv venv --python 3.12 .venv-quant
uv pip install --python .venv-quant/bin/python -r examples/quantization/requirements.txt
```

Use `vllm_venv` for the generation, extraction and deployment servers, and `.venv-quant` for the clients and quantization scripts. Run all commands from the repository root.

The requirements install speculators from this checkout, including its local hidden-state connector. Both scripts expect an unquantized DSpark checkpoint and a fresh output directory. The NVFP4 example replaces llm-compressor 0.15.0's validator because Transformers `AutoConfig` cannot read the speculators config. This replacement accepts only DSpark and rejects quantized and MTP checkpoints.

Passing a drafter Hub ID to either quantization script loads its default revision; the scripts do not pin it. To use a particular drafter revision, download it separately and pass its local directory as `--model` (the `DRAFT` value below). Record the revision used.

## Generate calibration data

FP8 does not need data. For NVFP4, regenerate responses for representative prompts and extract verifier hidden states for at least 1024 eligible rows. The number of eligible rows can be smaller than `--limit`; inspect the output and increase the limit if needed. Full-length 2048-token BF16 caches can require approximately 180 GiB of disk for 1024 rows, plus staging and checkpoint space. During quantization, llm-compressor's basic pipeline caches all calibration inputs in host RAM: eight layers of width 5120 at 1024 × 2048 tokens require approximately 160 GiB for BF16 hidden states alone, plus other inputs and overhead. Shorter rows need less. See the [response regeneration guide](../../docs/user_guide/tutorials/response_regeneration.md) for dataset preparation.

Set these variables in both the server and client terminals, from the repository root:

```bash
MODEL=Qwen/Qwen3.8-27B
DRAFT=RedHatAI/Qwen3.8-27B-speculator.dspark
RUN="$PWD/local/dspark-calibration-1024"
SAMPLES=1024
mkdir -p "$RUN"
```

In the server terminal, activate the vLLM environment and start a generation server. Record its command, model revision, versions and sampling parameters in `$RUN`:

```bash
source vllm_venv/bin/activate
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8000/v1/models` succeeds, activate the quantization environment in the client terminal and prepare rows. The minimum of nine supervised tokens ensures an anchor before the final eight-token block. Confirm the printed count is at least 1024 before continuing; if needed, increase `--limit` and prepare a fresh run.

```bash
source .venv-quant/bin/activate
speculators regenerate-responses --dataset ./representative-prompts.jsonl \
  --model "$MODEL" --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit 1280 --concurrency 4 --max-tokens 1024 \
  --sampling-params '{"temperature":0.6,"top_p":0.95,"seed":42}' \
  --outfile "$RUN/responses.jsonl" --resume
speculators prepare-data --model "$MODEL" --data "$RUN/responses.jsonl" \
  --output "$RUN/prepared" --max-samples "$SAMPLES" --seq-length 2048 \
  --minimum-valid-tokens 9 --seed 42
python -c 'import sys; from datasets import load_from_disk; print(len(load_from_disk(sys.argv[1])))' "$RUN/prepared"
```

Stop the generation server. In the same vLLM server terminal, start an extraction server on a shared filesystem. The launcher appends verifier layer 64 to the eight draft layers.

```bash
python scripts/launch_vllm.py train "$MODEL" \
  --target-layer-ids 4 12 20 28 36 44 52 60 \
  --hidden-states-path "$RUN/staging" --provenance-dir "$RUN" \
  -- --host 127.0.0.1 --port 8001 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8001/v1/models` succeeds, run the client in the quantization terminal:

```bash
speculators generate-offline-data --model "$MODEL" \
  --endpoint http://127.0.0.1:8001/v1 --preprocessed-data "$RUN/prepared" \
  --output "$RUN/hidden_states" --max-samples "$SAMPLES" --concurrency 4 \
  --validate-outputs --fail-on-error
```

Stop the server before quantization. Use a fresh cache for each set of rows and extraction settings. The adapter validates token alignment, shape, finite values and layer order; it accepts FP16, BF16 and FP32 states, not raw scaled FP8. Check that the cache belongs to `vllm_command.txt`; provenance is not cryptographically bound to the cache.

## Quantize and deploy

Run quantization in the `.venv-quant` client terminal:

```bash
python examples/quantization/dspark_fp8.py --model "$DRAFT" \
  --output-dir "$RUN/draft-fp8"
python examples/quantization/dspark_nvfp4_gptq_imatrix.py --model "$DRAFT" \
  --data-dir "$RUN/prepared" --hidden-states-dir "$RUN/hidden_states" \
  --extraction-provenance "$RUN/vllm_command.txt" --samples "$SAMPLES" \
  --output-dir "$RUN/draft-nvfp4"
```

The NVFP4 example runs whole-model calibration. The extraction command's verifier argument must exactly match the verifier name in the drafter config; a local snapshot path or alias for `MODEL` will be rejected even if it contains the same weights. The drafter also loads verifier weights by that name. Ensure it resolves to the same verifier revision used for generation and extraction; the script checks the name, not the revision. Record the revisions, package and CUDA versions, prompt source and checksum, commands, seed, selected rows and recipe alongside the outputs. Keep the launcher provenance and patches with the run.

In the `vllm_venv` server terminal, serve each output separately with the matching verifier and a fresh provenance directory:

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
