# Quantize a DSpark drafter

These examples target `RedHatAI/Qwen3.8-27B-speculator.dspark` with verifier `Qwen/Qwen3.8-27B`. FP8 uses model-free quantization; NVFP4 uses GPTQ with iMatrix calibration on verifier hidden states. Both leave embeddings, vocabulary, Markov and confidence heads unquantized.

Create a Python 3.12 environment from the repository root and install the examples' dependencies. Install a PyTorch build appropriate for your hardware.

```bash
uv venv --python 3.12 .venv-quant
uv pip install --python .venv-quant/bin/python . ./hs_connectors -r examples/quantization/requirements.txt
uv pip check --python .venv-quant/bin/python
source .venv-quant/bin/activate
```

The NVFP4 example uses an experimental DSpark-only override of llm-compressor's validator because the released validator cannot read this checkpoint's speculators config. It rejects quantized and MTP checkpoints and checks the validator fingerprint. Both scripts require a new output directory and an unquantized DSpark checkpoint. The default draft revision is `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce`; override it with `--revision` for another Hub commit.

## Generate calibration data

FP8 does not need data. For NVFP4, regenerate responses for at least 1024 representative prompts and extract their verifier hidden states. The number of eligible rows can be smaller than `--limit`; inspect the output and increase the limit if needed. Full-length 2048-token BF16 caches can require approximately 180 GiB for 1024 rows, plus staging and checkpoint space. See the [response regeneration guide](../../docs/user_guide/tutorials/response_regeneration.md) and [training tutorial](../../docs/user_guide/tutorials/train.md) for server setup.

```bash
MODEL=Qwen/Qwen3.8-27B
DRAFT=RedHatAI/Qwen3.8-27B-speculator.dspark
RUN="$PWD/local/dspark-calibration-1024"
SAMPLES=1024
mkdir -p "$RUN"
```

Start a vLLM generation server, and record its command, model revision, versions and sampling parameters in `$RUN`:

```bash
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8000/v1/models` succeeds, regenerate responses and prepare rows. The minimum of nine supervised tokens ensures an anchor before this draft's final eight-token block, including after truncation.

```bash
speculators regenerate-responses --dataset ./representative-prompts.jsonl \
  --model "$MODEL" --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit 1280 --concurrency 4 --max-tokens 1024 \
  --sampling-params '{"temperature":0.6,"top_p":0.95,"seed":42}' \
  --outfile "$RUN/responses.jsonl" --resume
speculators prepare-data --model "$MODEL" --data "$RUN/responses.jsonl" \
  --output "$RUN/prepared" --max-samples "$SAMPLES" --seq-length 2048 \
  --minimum-valid-tokens 9 --seed 42
python - "$RUN/prepared" "$SAMPLES" <<'PY'
import sys
from datasets import load_from_disk
count = len(load_from_disk(sys.argv[1]))
print(f"Prepared {count} rows; need {sys.argv[2]}")
if count < int(sys.argv[2]):
    raise SystemExit("Not enough prepared rows")
PY
```

Stop the generation server. In a vLLM environment with `hs_connectors` installed, start an extraction server on a shared filesystem. The launcher appends verifier layer 64 to the eight draft layers.

```bash
python scripts/launch_vllm.py train "$MODEL" \
  --target-layer-ids 4 12 20 28 36 44 52 60 \
  --hidden-states-path "$RUN/staging" --provenance-dir "$RUN" \
  -- --host 127.0.0.1 --port 8001 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8001/v1/models` succeeds:

```bash
speculators generate-offline-data --model "$MODEL" \
  --endpoint http://127.0.0.1:8001/v1 --preprocessed-data "$RUN/prepared" \
  --output "$RUN/hidden_states" --max-samples "$SAMPLES" --concurrency 4 \
  --validate-outputs --fail-on-error
```

Stop the server before quantization. Use a fresh cache for each set of rows and extraction settings. The adapter validates token alignment, shape, finite values and layer order; it accepts FP16, BF16 and FP32 states, not raw scaled FP8. Check that the cache belongs to `vllm_command.txt`; provenance is not cryptographically bound to the cache.

## Quantize and deploy

```bash
python examples/quantization/dspark_fp8.py --model "$DRAFT" \
  --output-dir "$RUN/draft-fp8"
python examples/quantization/dspark_nvfp4_gptq_imatrix.py --model "$DRAFT" \
  --data-dir "$RUN/prepared" --hidden-states-dir "$RUN/hidden_states" \
  --extraction-provenance "$RUN/vllm_command.txt" --samples "$SAMPLES" \
  --sequence-length 2048 --max-anchors 32 --seed 42 \
  --output-dir "$RUN/draft-nvfp4"
```

The NVFP4 example runs whole-model calibration and can require substantial memory. The drafter loads verifier weights by the name in its config: ensure that resolves to the same verifier revision used for generation and extraction. Record the revisions, package and CUDA versions, prompt source and checksum, commands, seed, selected rows and recipe alongside the outputs. Keep the launcher provenance and patches with the run.

Serve each output separately with the matching verifier and a fresh provenance directory:

```bash
python scripts/launch_vllm.py eval "$MODEL" --spec-model "$RUN/draft-fp8" \
  --spec-tokens 7 --spec-method dspark --provenance-dir "$RUN/eval-fp8" \
  -- --port 9145 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --enforce-eager --per-request-spec-decode-metrics summary --no-enable-prefix-caching
curl -fsS http://127.0.0.1:9145/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B","prompt":"Explain photosynthesis briefly.","max_tokens":64,"temperature":0}'
```

Check for nonempty output and accepted draft tokens in `metrics.speculative_decoding`, then repeat with `draft-nvfp4` and `eval-nvfp4`. Evaluate on a representative dataset before drawing quality conclusions.

On H100, llm-compressor 0.15.0 completed both scripts on the target checkpoint with 128 offline calibration rows, and both outputs reloaded and served a speculative completion. The served FP8 checkpoint was generated separately and had matching weight hashes. NVFP4 reported different fused input scales; an earlier H100 smoke with llm-compressor development commit `eac6972ca25f9e67abd8011aad5bb30ef9915e55` used BF16 activations and did not consume those scales. Native NVFP4 W4A4 execution and recovery quality were not measured.
