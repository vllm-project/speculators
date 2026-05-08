# Algorithm Overview

Speculators organizes its workflows around three major axes: **training scheme** (how a speculator model is produced), **serving integration** (how it is deployed), and **data preparation strategy** (how training data is sourced and shaped). Understanding which combination applies to your situation determines the tools and configuration paths you will use.

| Pattern | One-line Description |
|---|---|
| Eagle3 Training | Train a speculative decoder head using the Eagle3 algorithm against a frozen target LLM |
| vLLM Serve (config-driven) | Deploy a trained speculator directly in vLLM, with all speculative decoding parameters read from `speculators_config` |
| Offline Data Generation | Pre-generate hidden-state training data from a target model before training begins |
| Response Regeneration | Refresh or augment dataset responses using a running inference server |
| Convert & Export | Convert a trained speculator into the speculators format for direct vLLM compatibility |
| Evaluation with GuideLLM | Benchmark speculative decoding throughput and latency against a baseline |

---

## Algorithm Selection

Speculators supports multiple speculative decoding algorithms. While this guide focuses on Eagle3 (the primary training algorithm with full tooling support), understanding the alternatives helps you choose the right approach for your constraints.

### Supported Algorithms

| Algorithm | Best For | Computational Cost | Training Support | Draft Quality |
|-----------|----------|-------------------|------------------|---------------|
| **Eagle3** | Production deployments requiring maximum performance | High (tree-based speculation) | ✅ Full (this guide) | Highest |
| **Greedy** | Resource-constrained environments, rapid prototyping | Low (sequential prediction) | Limited | Moderate |
| **MLP** | Custom architecture research, specialized draft models | Medium (depends on depth) | Via extension | Variable |
| **EAGLE v1/v2** | Legacy compatibility, external research checkpoints | Medium-High | Via conversion | High |
| **HASS** | External checkpoint integration from HASS repository | Medium-High | Via conversion | High |
| **Independent** | Specialized use cases, custom speculation strategies | Variable | Via registry | Variable |

### When to Use Alternatives to Eagle3

**Consider Greedy if:**
- You have limited GPU memory or compute resources

- You need simple, predictable behavior for debugging and testing
- You're prototyping speculative decoding workflows and want fast iteration
- Your use case prioritizes minimal memory footprint over draft acceptance rate

**Consider MLP if:**
- You're researching novel speculator architectures
- You need specific layer configurations not provided by Eagle3
- You're integrating with existing MLP-based language models
- You want fine-grained control over draft model architecture

**Use conversion patterns if:**
- You have a speculator already trained in external repositories (EAGLE v1/v2, HASS)
- You're migrating from research code to production deployment
- You want to leverage community-trained models not originally in speculators format

### Trade-offs Summary

- **Eagle3**: Produces the highest quality draft tokens via tree-based speculation, leading to better acceptance rates and throughput gains. However, requires significant GPU memory during training for hidden-state extraction and has more complex hyperparameter tuning (tree depth, branching factors).

- **Greedy**: Minimal computational and memory overhead with straightforward sequential prediction. Easier to debug and faster to train. However, lower draft acceptance rates compared to tree-based methods and limited ability to recover from early prediction errors.

- **MLP**: Flexible architecture allowing custom design for specific model families or constraints. Modular and extensible. However, requires custom training setup and may need architecture tuning for optimal performance.

- **EAGLE v1/v2 & HASS**: Access to pre-trained models from research communities. Battle-tested algorithms with published results. However, requires conversion to speculators format and may have limited documentation for training new models.

### Recommendation

**For most production use cases, Eagle3 is recommended.** It provides the best balance of draft quality, training tooling, and vLLM integration. The rest of this guide assumes you are working with Eagle3 training and deployment workflows.

For implementation details of non-Eagle3 algorithms, see the API Reference. For converting external Eagle v1/v2 or HASS checkpoints, see Pattern 5: Convert & Export below.

---

## Pattern Details

### 1. Eagle3 Training

**When to use it:** You have a target LLM (e.g., `meta-llama/Llama-3.1-8B-Instruct`) and want to attach a speculative decoder head that can propose multiple draft tokens per step, accelerating inference without changing the target model's weights.

**How it works:** Eagle3 is the primary algorithm for training speculative decoders within Speculators [[source: examples/data_generation_and_training/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/data_generation_and_training/README.md). The training pipeline consumes hidden-state activations extracted from the target model and trains a draft network to minimize the KL divergence between draft and target logit distributions over the vocabulary. The trained artifact includes a `speculators_config` block in its `config.json`, which encodes all speculative decoding hyperparameters needed at serve time [[source: examples/convert/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/convert/README.md).

**How it differs from alternatives:** Eagle3 is the primary algorithm variant for training within Speculators [[source: examples/data_generation_and_training/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/data_generation_and_training/README.md). The library also supports EAGLE (v1/v2/HASS), MLP, and Independent speculator types through its registry system. Future releases may expand training support for additional algorithm variants given the library's stated goal of being a unified library for speculative decoding algorithms.

**Trade-offs:**
- Training requires access to the full target model to extract hidden states, which demands significant GPU memory.
- The resulting speculator is tightly coupled to the specific target model architecture and checkpoint.
- Once trained, the speculator is self-describing: the `speculators_config` removes the need for manual parameter tuning at serve time [[source: examples/convert/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/convert/README.md).

---

### 2. vLLM Serve (Config-Driven Deployment)

**When to use it:** You have a trained speculator model (either from Speculators training or a pre-trained model from a hub such as `RedHatAI/Qwen3-8B-speculator.eagle3`) and want to run it in production with minimal configuration overhead.

**How it works:** All models trained through Speculators include a `speculators_config` in their `config.json`. These models are in the speculators format and directly runnable in vLLM using `vllm serve </path/to/speculator/model>`, which will apply all the speculative decoding parameters defined in the `speculators_config` [[source: examples/convert/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/convert/README.md). The quickstart command:

```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

runs the model in vLLM using default arguments defined in the `speculators_config` of the model's `config.json` [[source: README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/README.md).

**How it differs from alternatives:** Manual vLLM speculative decoding configuration requires passing algorithm flags and draft model paths explicitly on the command line. The config-driven pattern encapsulates all of this inside the model artifact, making deployment a single command regardless of the underlying algorithm [[source: examples/convert/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/convert/README.md).

**Trade-offs:**
- Simplicity comes at the cost of flexibility: overriding individual speculative decoding parameters requires either modifying `config.json` or passing explicit vLLM flags that supersede the embedded config.
- The pattern assumes vLLM is the serving runtime; other runtimes would need to parse `speculators_config` independently. [INFERENCE]

---

### 3. Offline Data Generation

**When to use it:** You want to pre-compute the hidden-state activations and token sequences that the Eagle3 draft head will train on, decoupling the expensive target-model forward passes from the training loop itself.

**How it works:** The `scripts/data_generation_offline.py` script passes data through the target (verifier) model using vLLM as its inference backend and writes the resulting training tensors to an output directory. This process requires the `datagen` optional install: `pip install speculators[datagen]` [[source: examples/data_generation_and_training/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/data_generation_and_training/README.md). A minimal invocation targets a model and a dataset:

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --train-data-path sharegpt \
    --output-dir ./training_data \
    --max-samples 5000
```

For larger models, tensor parallelism and custom layer extraction are supported [[source: scripts/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/scripts/README.md):

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-70B-Instruct \
    --train-data-path ./my_data.jsonl \
    --seq-length 4096 \
    --hf-cache-dir ./cache \
    --output-dir ./training_data \
    --layer-ids 2 28 54 \
    --tensor-parallel-size 4
```

**How it differs from alternatives:** Online data generation (running the target model during training) avoids the storage overhead of pre-computed activations but couples GPU utilization for inference and training, often reducing overall throughput. Offline generation allows the two phases to run on different hardware or at different times.

**Trade-offs:**
- Requires substantial disk space for pre-computed activations, especially at large sequence lengths or sample counts.
- The dataset is static; if the training distribution needs to change, data generation must be re-run.
- Enables reproducible training runs from a fixed dataset snapshot.

---

### 4. Response Regeneration

**When to use it:** Your source dataset contains responses that are stale, low-quality, or mismatched to the target model's style, and you want to refresh them using a live inference server before generating hidden-state training data.

**How it works:** The `scripts/response_regeneration/` tooling connects to a running inference server and rewrites dataset responses. The `run_all.sh` orchestration script handles model serving and regeneration together [[source: scripts/response_regeneration/README.md]](https://github.com/vllm-project/speculators/blob/main/scripts/response_regeneration/README.md):

```bash
./run_all.sh --model "meta-llama/Llama-3.3-70B-Instruct" --dataset magpie
```

Individual script options support dataset selection, row limits, resumption from a previous run, and concurrency tuning [[source: scripts/response_regeneration/README.md]](https://github.com/vllm-project/speculators/blob/main/scripts/response_regeneration/README.md).

**How it differs from alternatives:** Using the raw dataset directly (without regeneration) is faster but may produce a speculator trained on responses that do not match the target model's actual output distribution, degrading draft acceptance rates. [INFERENCE] Response regeneration is most valuable when using community datasets whose responses were generated by a different model.

**Trade-offs:**
- Requires a running inference server, adding operational complexity.
- Regeneration is time-consuming for large datasets.
- Produces a dataset whose responses are aligned to the specific target model checkpoint, improving speculator quality.

---

### 5. Convert & Export

**When to use it:** You have a speculator trained in external speculative decoding repositories (e.g., Eagle, HASS), or you need to ensure a model artifact conforms to the speculators format before serving it in vLLM.

**How it works:** The conversion tooling targets models trained in external speculative decoding repositories such as Eagle and HASS [[source: docs/convert.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/docs/convert.md). The entry points are the `convert` CLI command and `convert_model` Python API. All models in the speculators format carry a `speculators_config` in `config.json` and are directly runnable via `vllm serve </path/to/speculator/model>` [[source: examples/convert/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/convert/README.md). Conversion aligns the model's metadata with the expected schema so that vLLM can read speculative decoding parameters without additional flags.

**Trade-offs:**
- Conversion is a one-time cost that pays dividends in deployment simplicity.
- Models not converted to the speculators format can still be served in vLLM but require manual parameter specification at the command line.

---

### 6. Evaluation with GuideLLM

**When to use it:** You want to measure the throughput and latency impact of speculative decoding relative to a standard serving baseline, using a reproducible benchmark harness.

**How it works:** The `examples/evaluate/eval-guidellm/` directory provides environment-file-driven evaluation configs for specific model/algorithm combinations. Setup and execution follow a two-step pattern [[source: examples/evaluate/eval-guidellm/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/evaluate/eval-guidellm/README.md):

```bash
bash setup.sh  # or: bash setup.sh --use-uv for faster installation
./run_evaluation.sh -c configs/llama-3.1-8b-eagle3.env
```

Pre-built configs exist for Llama-3.1-8B Eagle3, Llama-3.3-70B Eagle3, GPT-OSS-20B Eagle3, Qwen3-8B Eagle3, and Qwen3-32B Eagle3 [[source: examples/evaluate/eval-guidellm/README.md]](https://github.com/vllm-project/speculators/blob/v0.3.0/examples/evaluate/eval-guidellm/README.md).

**Trade-offs:**
- Environment-file configs make benchmarks reproducible and shareable but require updating when model paths or serving parameters change.
- GuideLLM evaluation measures end-to-end serving performance; it does not isolate draft acceptance rate or per-layer latency independently. [INFERENCE]

---

## Decision Guide

Use the following table to select the appropriate pattern for your situation:

| Situation | Recommended Pattern |
|---|---|
| You have a pre-trained speculator from the hub and want to serve it immediately | **vLLM Serve (config-driven)** |
| You want to train a new speculator for a target LLM | **Offline Data Generation → Eagle3 Training** |
| Your training dataset responses don't match the target model's style | **Response Regeneration → Offline Data Generation → Eagle3 Training** |
| You have a speculator trained outside Speculators and want vLLM compatibility | **Convert & Export → vLLM Serve** |
| You want to measure the performance benefit of speculative decoding | **Evaluation with GuideLLM** |
| You are contributing to Speculators and need the development version | `git clone` + `pip install -e .` (see Getting Started) |

### Full Training Pipeline Sequence

For users building a speculator from scratch, the canonical end-to-end sequence is:

1. **(Optional)** Run response regeneration to align dataset responses to the target model.
2. Run offline data generation to extract hidden-state activations.
3. Train the Eagle3 speculator head using the generated data.
4. The trained artifact is automatically in speculators format with an embedded `speculators_config`.
5. Serve directly with `vllm serve <path/to/speculator>`.
6. Evaluate throughput gains with GuideLLM.

For implementation details of each step, see the How-To Guides. For configuration options embedded in `speculators_config`, see the API Reference.

---

## Version and Build Provenance

Speculators uses a dynamic versioning scheme driven by git history and environment variables. The build system supports five build types: `release`, `candidate`, `nightly`, `alpha`, and `dev` [[source: setup.py]](https://github.com/vllm-project/speculators/blob/v0.3.0/setup.py). Release builds append a post-release suffix if there are commits since the last tag (defaulting to the number of commits as the iteration); all pre-release types increment to the next minor version and append the appropriate PEP 440 qualifier (`rc`, `a`, or `dev`) [[source: setup.py]](https://github.com/vllm-project/speculators/blob/v0.3.0/setup.py).

Version metadata is baked into the installed package via `version.txt` and `version.py` at build time [[source: setup.py]](https://github.com/vllm-project/speculators/blob/v0.3.0/setup.py), allowing runtime introspection via the CLI's `--version` flag [[source: src/speculators/__main__.py:40-53]](https://github.com/vllm-project/speculators/blob/v0.3.0/src/speculators/__main__.py#L40-L53) without requiring access to the VCS. A key trade-off is that the generated files are written to the source tree, which can produce a dirty working-tree state during development [[source: rationale]](https://github.com/vllm-project/speculators/blob/v0.3.0/rationale). In CI environments with shallow git clones, the version computation falls back to `LAST_RELEASE_VERSION`, which may produce misleading version strings if not explicitly overridden [[source: rationale]](https://github.com/vllm-project/speculators/blob/v0.3.0/rationale).