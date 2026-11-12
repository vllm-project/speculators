# DSpark quantization examples

These examples target `RedHatAI/Qwen3.8-27B-speculator.dspark` with verifier `Qwen/Qwen3.8-27B`. FP8 uses dynamic activations without calibration; NVFP4 uses GPTQ and iMatrix on verifier-generated hidden states. Both exclude embeddings, vocabulary heads, Markov and confidence heads. A 128-sample H100 smoke covered offline extraction, quantization and reload, and one speculative completion with each output. It did not measure recovery quality or native NVFP4 W4A4 execution.

The compressor integration is experimental and tied to llm-compressor commit `eac6972ca25f9e67abd8011aad5bb30ef9915e55`. On that revision, the default validator cannot load a speculators config. The examples use a DSpark-only override that reads the original local checkpoint config and rejects quantized and MTP inputs. It relies on compressor internals, and checks the validator's fingerprint at runtime; an unreviewed validator fails closed. An upstream speculator-aware validator is preferable.

A Python 3.12 quantization environment with the versions below ran both examples on the target checkpoint and reloaded both outputs. The smoke used one H100 80 GB, PyTorch 2.14.1+cu130, and verifier revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. A separate vLLM 0.33.0 environment ran extraction and serving with eager execution. Use a separate vLLM environment with `hs_connectors` installed for extraction; see the [training tutorial](../../docs/user_guide/tutorials/train.md).

From the speculators checkout, create a fresh quantization environment and install the pinned compressor checkout and dependencies:

```bash
git clone https://github.com/vllm-project/llm-compressor.git /tmp/dspark-llm-compressor
git -C /tmp/dspark-llm-compressor checkout eac6972ca25f9e67abd8011aad5bb30ef9915e55
uv venv --python 3.12 .venv-quant
uv pip install --python .venv-quant/bin/python . ./hs_connectors /tmp/dspark-llm-compressor \
  'compressed-tensors==0.20.2a20261103' 'transformers==5.19.0' 'datasets==5.1.0' 'torch==2.14.1'
uv pip check --python .venv-quant/bin/python
source .venv-quant/bin/activate
```

Install the appropriate PyTorch build for your hardware and record the actual installed versions. Do not substitute a different verifier or disable compressor validation to work around a failure.

## Generate calibration data

Choose prompts representative of your deployment. This example regenerates responses using a local prompt JSONL supported by `regenerate-responses`; see the [response regeneration guide](../../docs/user_guide/tutorials/response_regeneration.md). Use a fresh run directory for each dataset and cache. Here 1024 is the number of eligible prepared samples; 128 is suitable for a local smoke test, not a measured recovery recipe. Individual full-length 2048-token BF16 caches can take about 180 MiB each, or roughly 180 GiB for 1024, plus staging and checkpoint space. Actual sizes depend on sequence lengths. The 128-row smoke cache occupied 3.2 GiB; all responses had been cut at 64 generated tokens, so it was not a representative calibration dataset.

```bash
MODEL=Qwen/Qwen3.8-27B
DRAFT=RedHatAI/Qwen3.8-27B-speculator.dspark
RUN="$PWD/local/dspark-calibration-1024"
SAMPLES=1024
LIMIT=1280
mkdir -p "$RUN"
```

Start a plain vLLM generation server in its own terminal. Record its command, model revision, versions and sampling parameters in `$RUN` because this direct vLLM command does not write launcher provenance.

```bash
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --max-model-len 2049
```

Wait for `curl -fsS http://127.0.0.1:8000/v1/models`. In another terminal, regenerate responses, then prepare rows. `--limit` counts input conversations, not successful output rows. Inspect the response/error files and increase the limit if too few rows survive. Responses and prepared rows can be truncated by the generation and sequence limits. The checkpoint uses block size 8, so `--minimum-valid-tokens 9` ensures each row has a supervised token before its last eight positions, even after truncation. This conservative filter can discard some otherwise usable rows. Keep the preparation and calibration sequence lengths equal; for another checkpoint, set the minimum to its block size plus one.

```bash
speculators regenerate-responses --dataset ./representative-prompts.jsonl \
  --model "$MODEL" --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit "$LIMIT" --concurrency 4 --max-tokens 1024 \
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

Stop the generation server and start the extraction server in the vLLM environment. The draft requires layers 4, 12, 20, 28, 36, 44, 52 and 60 in this order; the launcher appends final verifier layer 64. Use a shared filesystem for staging and the client. Extraction reserves one output token, so its server limit exceeds the prepared sequence length.

```bash
python scripts/launch_vllm.py train "$MODEL" \
  --target-layer-ids 4 12 20 28 36 44 52 60 \
  --hidden-states-path "$RUN/staging" --provenance-dir "$RUN" \
  -- --host 127.0.0.1 --port 8001 --max-model-len 2049
```

After `curl -fsS http://127.0.0.1:8001/v1/models` succeeds, extract the states:

```bash
speculators generate-offline-data --model "$MODEL" \
  --endpoint http://127.0.0.1:8001/v1 --preprocessed-data "$RUN/prepared" \
  --output "$RUN/hidden_states" --max-samples "$SAMPLES" --concurrency 4 \
  --validate-outputs --fail-on-error
```

Stop the extraction server before quantization. The adapter reads the first `$SAMPLES` rows, validates token alignment, shape, finite values and layer order from `vllm_command.txt`, and fails on missing samples. It accepts FP16, BF16 and FP32 states, not raw scaled FP8. `--validate-outputs` applies to newly generated files; never reuse a cache with changed rows or extraction settings. Verify the cache really belongs to the supplied provenance: the launcher cannot bind that file cryptographically to the extracted data.

## Quantize

```bash
python examples/quantization/dspark_fp8.py --model "$DRAFT" \
  --output-dir "$RUN/draft-fp8"
python examples/quantization/dspark_nvfp4_gptq_imatrix.py --model "$DRAFT" \
  --data-dir "$RUN/prepared" --hidden-states-dir "$RUN/hidden_states" \
  --extraction-provenance "$RUN/vllm_command.txt" --samples "$SAMPLES" \
  --sequence-length 2048 --max-anchors 32 --seed 42 \
  --output-dir "$RUN/draft-nvfp4"
```

The default draft revision is `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce`; use `--revision` to select another immutable Hub commit. Output directories must not already exist. The NVFP4 example uses the basic whole-model calibration pipeline, which can use substantial memory. It uses a loss-free backbone forward and avoids padded anchors. Exact numerical results depend on kernels and hardware; these settings are starting points, not independently verified quality results.

Record the immutable draft and verifier revisions, actual packages and CUDA versions, prompt source and checksum, commands, seed, selected rows, sample count, sequence length, anchor count and recipe alongside the result. The draft loader also reads verifier weights by the verifier name in its config, so ensure that name resolves to the same verifier revision used for generation and extraction; this is not automatically verified. `vllm_command.txt`, `vllm.patch`, and `checkpoint_sha256.txt` record extraction provenance on a best-effort basis, but do not record the entire workflow.

## Deployment smoke

Serve each output separately with the matching verifier. The H100 smoke used eager execution; use a fresh provenance directory per checkpoint:

```bash
python scripts/launch_vllm.py eval "$MODEL" --spec-model "$RUN/draft-fp8" \
  --spec-tokens 7 --spec-method dspark --provenance-dir "$RUN/eval-fp8" \
  -- --port 9145 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --enforce-eager --per-request-spec-decode-metrics summary --no-enable-prefix-caching
```

Send a completion request and check for nonempty output and accepted draft tokens in `metrics.speculative_decoding`:

```bash
curl -fsS http://127.0.0.1:9145/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B","prompt":"Explain photosynthesis briefly.","max_tokens":64,"temperature":0}'
```

Repeat with `draft-nvfp4` and `eval-nvfp4`, then run the repository acceptance evaluator on a representative dataset. In the H100 smoke, both outputs produced one 64-token completion with accepted draft tokens. vLLM reported 53.49 GiB for FP8 model loading and 52.74 GiB for NVFP4 model loading (verifier and drafter together, not total peak usage). The FP8 run selected `CutlassFP8ScaledMMLinearKernel`; NVFP4 selected `HummingNvFp4LinearKernel`. The NVFP4 loader warned that fused parallel layers had different input global scales. The tested H100 Humming configuration used BF16 activations and did not consume those scales; their impact on native FP4 execution remains unmeasured. Retain the server logs, software revisions, hardware, kernel/backend and provenance artifacts. Native NVFP4 requires suitable hardware; this H100 smoke does not demonstrate native W4A4 execution or recovery quality.
