# Agentic Regeneration with Verifiers

Agentic regeneration should have a narrow integration boundary:

1. the upstream Verifiers CLI runs an installed taskset and writes `config.toml` plus `traces.jsonl`;
2. one generic adapter converts each native root-to-leaf trace branch to exact `input_ids` and `loss_mask`;
3. Speculators replays those IDs through the same checkpoint to collect aligned hidden states.

There is no Speculators environment wrapper or rollout loop.

## Pinned UV environment

The adapter has its own UV project under `scripts/agentic_regeneration/`. Its small `pyproject.toml` pins the compatibility boundary: Python 3.12, the stable `verifiers==0.2.0` release, and one exact research-environments commit.

```bash
uv sync --project scripts/agentic_regeneration
```

This environment is intentionally separate from the main Speculators environment. Updating Verifiers or an upstream taskset is an explicit `pyproject.toml` change, while UV resolves ordinary transitive dependencies at install time.

## Environment profiles

Three native CLI profiles provide a useful progression:

| Profile | What it exercises | Cost |
|---|---|---|
| `prolog.toml` | Multi-turn code editing, execution feedback, and hidden verification | Public SWI-Prolog image; recommended smoke test |
| `livecodebench.toml` | Python code generation with hidden execution tests | Shared Python image; useful correctness baseline |
| `r2e.toml` | Multi-turn inspection and repair of a real Python repository | Per-task repository image; production acceptance test |

Prolog is the quickest genuinely agentic check: tasks are generated locally and the public sandbox image is small. LiveCodeBench is lighter coding data but is capped at one turn in the supplied profile. R2E-Gym is the target for realistic Python repair, and its image and dataset cost is inherent to reproducibly executing arbitrary repositories.

All three tasksets and the coding harness are upstream. R2E uses Verifiers' built-in stable `default` harness, which supplies bash and edit tools; Speculators contains no R2E-specific code.

## 1. Start Qwen3-8B in vLLM

```bash
python scripts/launch_vllm.py Qwen/Qwen3-8B \
  --hidden-states-path output/agentic_regen/server_hidden_states \
  -- \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 16384
```

The profiles use Verifiers' `train` client and its Qwen3 renderer with `enable_thinking = false`. Rendering and tool-call parsing therefore happen client-side, while vLLM only serves exact token generation. Disabling thinking keeps this coding profile from spending its whole turn on an unparsed reasoning block. OpenAI chat-completions tool-parser flags are not part of this path.

## 2. Run an upstream taskset

Start with Prolog:

```bash
VLLM_API_KEY=EMPTY uv run --project scripts/agentic_regeneration \
  eval @ scripts/agentic_regeneration/configs/prolog.toml
```

Switch only the final profile path for another environment:

```text
scripts/agentic_regeneration/configs/livecodebench.toml
scripts/agentic_regeneration/configs/r2e.toml
```

The stable CLI owns dataset loading, Docker lifecycle, tools, turns, retries, and scoring. Each profile is a one-task smoke run; increase `num_tasks`, `num_rollouts`, and `max_concurrent` in TOML for collection.

The `train` client is important: its native trace contains the exact rendered prompt and sampled completion token spans. A normal chat-completions response is not sufficient for exact hidden-state alignment.

## 3. Convert native traces

For the Prolog profile:

```bash
uv run --project scripts/agentic_regeneration \
  python scripts/agentic_regeneration/convert_traces.py \
  --traces output/agentic_regen/prolog/traces.jsonl \
  --outfile output/agentic_regen/prolog/trajectories.jsonl
```

The adapter targets the stable Verifiers `WireTrace` format. It reconstructs every root-to-leaf branch, retains messages and tool-call linkage, and reads model, endpoint, and taskset metadata from the sibling `config.toml`. Stable 0.2.0 does not duplicate tool schemas in the JSON trace; exact-token replay does not need to render them again.

Completed zero-reward traces are retained because unsuccessful actions are valid on-policy data.

## 4. Prepare and replay exact IDs

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data output/agentic_regen/prolog/trajectories.jsonl \
  --seq-length 16384 \
  --minimum-valid-tokens 1 \
  --num-preprocessing-workers 1 \
  --output output/agentic_regen/prolog/preprocessed

python scripts/data_generation_offline.py \
  --model Qwen/Qwen3-8B \
  --endpoint http://127.0.0.1:8000/v1 \
  --preprocessed-data output/agentic_regen/prolog/preprocessed \
  --output output/agentic_regen/prolog/eagle_data \
  --concurrency 1 \
  --validate-outputs \
  --fail-on-error
```

When a record contains both `input_ids` and `loss_mask`, preprocessing passes them through together rather than applying the chat template again. `--validate-outputs` then checks that vLLM returned those same IDs and an aligned hidden-state sequence.

The resulting states are on-policy for the served checkpoint because the model sampled every assistant turn against real observations, and replay teacher-forces that exact branch through the same weights. Regenerate after a policy update.
