# Modular speculative decoding: implementation plan

Status: implementation in progress. The first implementation checkpoint covers numerical references, backbone composition, checkpoint weight formats, and optimizer identity validation. Unchecked items remain future work; see the progress section below.

## Implementation progress

Implemented in separate Jujutsu changes:

- `test(models): capture legacy DFlash family contracts`: fixed-weight numerical references from `988ed16b1f91315942049da3093d88dba70c45d9`, covering all four existing recipes and both slot conventions. The fixture includes parameter keys, freezing/save ownership, optimizer partitioning, output probes, losses, metrics, and gradient probes.
- `refactor(models): compose DFlash backbones and expose decoder layers`: typed block layouts and backbone inputs/features, a target-free `draft_features()` path, explicit DFlash2 decoder construction, migration of all four forward callers, and a `get_decoder_layers()` training interface. Invalid short-input padding indices are now safe, and zero requested anchors are explicitly rejected.
- `fix(checkpoints): map composed DFlash weights at serialization boundaries`: flat serving exports remain the default for named recipes; `save_pretrained(..., checkpoint_format="native")` writes nested names with `checkpoint_format_version=1`. Legacy/native imports, sharded pretrained loads, converters, and direct checkpointer weight loading share the mapping and validate learned weights. Export does not mutate the live model config or consume the caller's tensor dictionary.
- `fix(train): validate optimizer identities when resuming composed models`: new checkpoints write `optimizer_metadata.json`. Named legacy single-device/DDP and distributed states can map old parameter names when group ordering matches. Unnamed legacy optimizer files without metadata fail explicitly; use `--from-pretrained` for weights-only initialization. Model-weight compatibility does not imply arbitrary legacy optimizer-state compatibility.

The current recipes remain:

| Recipe  | Backbone                                 | Correction / auxiliary modules              | Training behavior                                                          |
| ------- | ---------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| DFlash  | DFlash                                   | No correction                               | Distillation over anchored blocks                                          |
| DFlash2 | DFlash with convolutional decoder layers | Candidate selector                          | Unary distillation plus candidate reranking objective                      |
| DSpark  | DFlash                                   | Markov correction; optional confidence head | Corrected distillation plus optional confidence supervision                |
| XPress  | DFlash                                   | Iterative refiner                           | Teacher forcing, free-running consistency rounds, and base-logit anchoring |

Both anchor conventions remain configurable. Existing defaults, full/reduced vocabulary ownership, and attention/position alignment are retained. Correction modules and training objectives still live in the named algorithms; extracting them and enabling the first configurable mixed recipe is the next implementation stage. The interfaces currently cover the anchored-block family, not autoregressive inference state or arbitrary cross-family composition.

Validation includes CPU numerical parity, Dynamo tracing with the eager backend under changing valid-anchor counts, gradient-checkpointing discovery, FSDP wrapping wiring, flat/native sharded checkpoint round trips, malformed checkpoint rejection, and identical next optimizer updates after single-device save/resume. Actual GPU execution, multi-GPU FSDP resume, and serving-engine loading are still required before claiming those execution paths are validated. Preserving exported keys alone is not a serving integration test.

### Validation recorded for the first checkpoint

- The affected model, conversion, training/checkpointer, optimizer, and public model/config test selection passed with 295 passed and 10 GPU skips. The serialization/optimizer subset passed with 40 tests, including custom pretrained key mapping validation.
- The expanded selection initially produced 763 passed, 10 skipped, and 8 failures. One was the pre-existing `test_minimax_outer_model_type` assertion (also reproduced against the reference implementation); three required an uncached Hugging Face configuration in offline mode; three validation-loop tests attempted CUDA transfers without an available GPU; and one local Gloo test was blocked by sandbox socket restrictions. The Gloo test passed when rerun outside the sandbox. No production behavior was changed to accommodate those unrelated failures.
- `make quality` passed with `MDFILES` restricted to the 63 tracked Markdown files. The default Makefile file discovery also includes ignored nested workspaces/environments and failed formatting an unrelated `.workspaces/pr-1388/.pytest_cache/README.md`; those ignored files were left untouched.

## 1. Goal and scope

Allow backbones and correction modules to be configured independently, while preserving the behavior and checkpoint usability of existing algorithms. Begin with DFlash, DFlash2, DSpark, and XPress. Establish an interface that can later accommodate EAGLE3, P-EAGLE, and MTP without requiring their migration in the first release.

Use composition for parameterized components. Treat serialized names as a separate compatibility contract. Do not use multiple inheritance to preserve parameter paths, and do not register the same module under both old and new paths.

The first milestone must demonstrate a combination that does not require a new algorithm subclass: a DFlash2 convolutional backbone with DSpark Markov correction and optional confidence prediction. Passing structural and numerical checks establishes that the combination can be trained; it does not establish decoding quality or interchangeability of independently trained weights.

## 2. Architectural decisions

### Component ownership

| Component             | Responsibility                                                                                                                  | Initial implementations                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Model wrapper         | Assemble components, own vocabulary mappings and verifier reconstruction, expose public training and serialization entry points | Existing named models delegating to shared assembly, plus a modular entry point                 |
| Input preparation     | Select anchors, construct block layout and token embeddings, construct masks and positions                                      | Anchored-block preparation                                                                      |
| Backbone              | Project verifier features, run decoder layers, normalize prediction hidden states                                               | DFlash, DFlash2 convolutional variant                                                           |
| Vocabulary projection | Convert normalized backbone states into base draft logits                                                                       | Existing `lm_head`, retained on wrapper                                                         |
| Correction            | Modify logits or score candidate tokens using backbone features and conditioning tokens                                         | Identity, DSpark Markov, DFlash2 selector, XPress refiner                                       |
| Confidence prediction | Produce acceptance-related scores, optionally sharing explicitly supplied correction features                                   | Existing DSpark confidence head                                                                 |
| Training objective    | Construct targets, supply teacher-forced conditioning, run refinement passes, compute losses and metrics                        | DFlash, DSpark, DFlash2 selector, XPress objectives                                             |
| Proposal policy       | Specify how serving produces tokens from model outputs                                                                          | Existing policies and explicit requirements for sequential, candidate, and iterative correction |

The training objective is a strategy owned by the wrapper, not a container for learned parameters. Inference must not require verifier targets or a training loss. Keep learned modules under the model even when the objective orchestrates repeated calls to them.

### Internal module layout

```text
model
├── embed_tokens
├── backbone
│   ├── fc
│   ├── hidden_norm
│   ├── layers
│   ├── rotary_emb
│   └── norm
├── lm_head
├── correction
├── confidence_head          # optional
├── verifier_norm            # reconstructed verifier-owned state
└── verifier_lm_head         # reconstructed verifier-owned state
```

Retain `embed_tokens`, `lm_head`, `t2d`, and `d2t` on the wrapper initially. Preserve existing full-vocabulary versus reduced-vocabulary ownership and freezing rules. Use `correction` rather than `head` to distinguish correction from the vocabulary projection. One correction strategy is supported initially; arbitrary chains of corrections are deferred because ordering and training semantics need separate design.

### Checkpoint policy

Native modular checkpoints use nested parameter paths and an explicit format version. Existing unversioned configurations select a known legacy recipe and legacy key mapping. Loading translates into the internal layout; exporting to a legacy runtime translates out of it.

Preserve the default published format for existing named recipes during migration. The modular entry point saves the versioned native format by default. Expose explicit native save and legacy serving export choices, with matching configuration and tensor names. Do not infer the output format from module attributes or mutate the live model configuration during export.

## 3. Phase A: capture the existing contracts

- [x] Compare the implementation against the pre-refactor revision (`qkkxxtrm-` in the current workspace) using `jj`; record the concrete revision used for reference artifacts.
- [x] Inventory all DFlash-family callers of `layers`, `fc`, `norm`, `hidden_norm`, `rotary_emb`, `_make_decoder_layer`, `_backbone_forward`, and anchor helpers. Include converters, trainer setup, distributed wrapping, and tests.
- [x] Capture tiny legacy configurations and parameter-key manifests for DFlash, DFlash2, DSpark, and XPress. Use generated local fixtures with no verifier download.
- [x] Define the existing recipes explicitly: backbone type, correction type, confidence settings, objective, vocabulary mode, slot convention, causality, and proposal behavior.
- [x] Capture numerical reference cases using fixed weights and fixed anchor selections, rather than relying on constructor RNG order remaining identical after composition.
- [x] Record base logits, corrected outputs or candidate scores, scalar loss, selected metrics, and gradients for each existing recipe. Include XPress refinement passes and DFlash2 convolution identity initialization.
- [x] Capture parameter ownership: trainable tensors, verifier-reconstructed tensors, omitted save keys, and optimizer groups.

Current review baseline: the focused utility, target, DFlash2 definition, and converter tests produced 42 passes, 8 failures, and 1 GPU skip with `HF_HUB_OFFLINE=1`. The failures include missing model-level `layers`, `fc`, and `rotary_emb` attributes. Direct checks also found missing `_backbone_forward` calls in DSpark and XPress and failure of DFlash training compatibility validation. These are migration tasks, not acceptable final exclusions.

Completion: reference cases can be reproduced locally and the legacy contracts are recorded independently of the new implementation.

## 4. Phase B: define typed component contracts

- [x] Add a small shared component-interface module, separate from `models/base_components.py`, which currently selects transformer layer, norm, and rotary classes rather than complete speculative-decoding components.
- [x] Define an anchored-block layout object carrying anchor positions, flattened source indices, block size, valid prediction mask, predecessor alignment, and slot convention. Keep document boundaries and source position IDs explicit where preparation needs them.
- [x] Document which tensors use verifier vocabulary IDs and which use draft vocabulary IDs. Preserve the existing `d2t` offset semantics rather than treating it as a direct lookup table.
- [ ] Define backbone inputs: prepared draft embeddings, captured verifier features, position information, attention masks, and optional inference state. Specify shapes and whether outputs are already normalized.
- [ ] Define backbone output as hidden states aligned to prediction slots, with optional explicitly typed inference state. Keep targets, logits, and losses out of the backbone output.
- [ ] Define correction inputs: normalized hidden states, base logits, conditioning token IDs, and block layout. Add an explicit entry point for candidate scoring so selector implementations need not materialize a corrected full-vocabulary tensor.
- [ ] Define correction outputs that distinguish corrected vocabulary logits from candidate IDs and scores. Define optional features consumed by confidence prediction without exposing Markov implementation internals to the wrapper.
- [ ] Keep correction evaluation free of implicit ground-truth access. The objective supplies teacher-forced predecessors during training; a proposal policy supplies generated predecessors during inference.
- [ ] Define the confidence input contract for hidden-only and hidden-plus-Markov-feature modes. Reject the latter when its required feature provider is absent.
- [ ] Define a training-objective interface that receives the assembled model and prepared batch and returns the existing trainer-compatible result (`None`, loss, metrics). Preserve current loss reduction and metric semantics during extraction.
- [x] Define `get_decoder_layers()` on the model wrapper for training infrastructure. Legacy models may initially implement it by returning their existing `layers`; modular models delegate to their backbone. Document that all returned layers are registered exactly once.
- [ ] Specify compatibility validation for hidden dimensions, vocabulary mappings, block size, slot alignment, candidate count, required correction features, and proposal policy. Validate combinations at construction, before training begins.
- [ ] Keep configuration validation outside the compiled tensor path. Avoid component-specific string dispatch or mutable Python configuration in per-token execution.

Completion: interfaces can express all four existing recipes and the first mixed recipe without algorithm-specific branches in the backbone.

## 5. Phase C: make the backbone extraction behavior-preserving

Primary files: `models/dflash/core.py`, `models/dflash/utils.py`, `models/dflash2/core.py`, `models/dflash2/model_definitions.py`, `model.py`, and `train/distributed.py` under `src/speculators/`.

- [x] Finish `DFlashBackbone` as a computation-only module. Correct its stale return-value docstring, add input/output annotations, and clarify normalized output semantics.
- [x] Move DFlash2 layer customization into an explicit backbone implementation or construction factory. Ensure convolution configuration reaches it without relying on a factory override on the outer model.
- [x] Preserve DFlash2 convolution reset behavior after model initialization. Ensure wrapper initialization does not initialize shared or frozen tensors twice or overwrite specialized identity initialization.
- [x] Extract anchored-block preparation from `DFlashDraftModel.forward()`. Preserve sorted anchors, padded-anchor masking, packed-document isolation, position resets, causality, sliding/full attention routing, and anchor-slot masking.
- [x] Keep target construction in the objective/preparation side, including shifted verifier logits and the selected-position optimization. Preserve verifier final-norm conventions.
- [x] Define behavior for no valid anchors, zero requested anchors if supported, and sequence length shorter than block size. Ensure padded blocks cannot trigger invalid source indexing even when their loss is zero.
- [x] Replace `model.layers` requirements in `verify_training_compatible()` and `apply_fully_sharded()` with `get_decoder_layers()`.
- [x] Audit gradient checkpointing discovery and FSDP no-split declarations for both decoder types. Preserve frozen-parameter replication and mixed-precision handling.
- [x] Migrate existing subclass callers to a shared prepared-feature method or the new assembly flow. If a temporary `_backbone_forward` compatibility wrapper is used, keep exactly one implementation of preparation and remove the wrapper after its internal callers migrate.
- [x] Preserve required attention kwargs explicitly; do not silently discard previously supported execution options during extraction.
- [ ] Verify dynamic anchor selection and attention-mask construction under `torch.compile`. Preserve intentional compiler-disabled boundaries or replace them only with measured evidence.

Completion: existing DFlash-family training forwards and trainer setup work with nested backbones before adding user-configurable combinations.

## 6. Phase D: implement checkpoint adapters across every loading path

Primary files: `config.py`, `model.py`, `train/checkpointer.py`, `train/trainer.py`, and `convert/dflash/converter.py` under `src/speculators/`.

- [x] Add a checkpoint format version to native modular model configuration. Keep training-state format metadata distinct from model-weight format metadata.
- [ ] Implement named adapters for each legacy recipe and external source format. Use anchored prefix mappings, not unrestricted substring replacement.
- [x] Centralize mapping and validation so all entry points share the same rules. Keep ordinary internal `state_dict()` names native and predictable.
- [x] Cover all entry points: `SpeculatorModel.from_pretrained()`, concrete model loading, explicit `state_dict` loading, converter loads, single-GPU/DDP resume, distributed state-dict loading, and best-checkpoint reloads.
- [x] Inspect the installed Transformers loading implementation before selecting its key-mapping integration. Verify the chosen mechanism actually applies to pretrained loading; do not assume ordinary `load_state_dict` hooks cover it.
- [x] Support sharded safetensors and their index files. Translate key metadata consistently and avoid collecting a second full model copy just to rename keys.
- [ ] Implement explicit import/export mappings for `layers.*`, `fc.*`, `hidden_norm.*`, `norm.*`, and each correction module's legacy keys. Map persistent buffers too; do not invent checkpoint entries for nonpersistent rotary buffers.
- [x] Reject duplicate source keys mapping to one destination, mixed ambiguous formats, unknown versions, shape mismatches, and missing learned backbone/correction weights. Restrict missing-key exceptions to documented verifier-owned or optional tensors.
- [x] Update save omission lists after renaming. Preserve full-vocabulary frozen LM-head reconstruction, reduced-vocabulary LM-head saving, vocabulary mappings, and verifier norm reconstruction.
- [x] Update DFlash converter accesses to `model.backbone.fc` and its validation keys. Map both ordinary and fused-QKV source weights, preserving projection conflict checks, FC slicing, and dropped-key reporting.
- [x] Ensure native save and legacy export work when supplied an already gathered state dict by the checkpointer. Keep serialized configuration, weight names, architecture identifier, and dtype consistent.
- [ ] Reject legacy exports of unsupported mixtures before writing partial output. Never label an arbitrary mixture as a known serving architecture just because shapes match.
- [ ] Add a deliberate one-time loader for this branch's intermediate nested layout only if an inventory finds checkpoints worth preserving; otherwise document that development-only layout as unsupported.

### Training-state migration

- [x] Inventory single-GPU optimizer parameter ordering and names, plus distributed optimizer fully qualified names. Module nesting can change parameter order even when tensors are identical.
- [x] For new training states, store enough versioned parameter identity/group metadata to validate or reconstruct optimizer assignments. Include multiple optimizers and schedulers.
- [x] Map legacy distributed optimizer names with the model-key adapter where a verified migration is possible.
- [ ] For legacy positional optimizer states, derive the old recipe's parameter ordering and validate identities/shapes and group membership. Do not assume matching tensor counts or shapes prove the assignment is correct.
- [x] If an old training state cannot be mapped reliably, fail full resume with a specific explanation and document the explicit model-weights-only initialization path. Never silently reset optimizer state during a requested resume.
- [ ] Preserve scheduler state, step counters and dtypes, epoch/local-step metadata, sampler position, and best-validation bookkeeping for supported resume paths. Preserve any existing RNG-state handling; do not claim exact stochastic continuation beyond what is restored.
- [x] Test one additional optimizer step after save/resume against uninterrupted execution with deterministic inputs and restored state.

Completion: legacy model weights load correctly, current recipes export correctly, native round trips work, and full resume is either tested or explicitly rejected for a documented legacy format.

## 7. Phase E: extract correction modules and training objectives

- [ ] Add identity correction for plain DFlash, with no learned parameters and no unnecessary full-logit copy.
- [ ] Adapt DSpark `MarkovHead` behind the correction interface, preserving vanilla, gated, and RNN modes. Keep predecessor lookup, recurrent state, and train/inference conditioning semantics explicit.
- [ ] Keep confidence prediction independently selectable. Preserve the existing shared Markov embedding path when configured, without recomputing features or detaching required gradients.
- [ ] Adapt DFlash2 `CandidateSelector` behind candidate scoring. Preserve top-k selection, strict versus replace miss policy, target injection during training, and candidate-only normalization/loss behavior.
- [ ] Adapt `XPressRefinerHead` behind an explicit refinement operation. Keep the objective responsible for teacher forcing, free-running consistency rounds, stop-gradient boundaries, and base-logit anchoring.
- [ ] Move existing per-algorithm forward orchestration into objective strategies using the shared preparation and backbone paths. Reuse existing metric implementations before attempting loss-code consolidation.
- [ ] Preserve `sample_from_anchor` behavior across correction conditioning, targets, valid-slot masking, per-position weighting, acceptance metrics, and speculative-token counts.
- [ ] Verify each extracted component's trainable parameters receive the intended gradients and frozen verifier-owned parameters remain frozen.
- [ ] Make component construction own initialization rules. Verify identity/no-op initializations and explicitly identify any intentional numerical behavior changes in separate work.
- [ ] Keep ordinary inference correction separate from training-only target injection and loss calculation.

Completion: each original named algorithm is expressed as a recipe over shared components and reproduces its numerical reference cases.

## 8. Phase F: configuration, factories, and user-facing assembly

Primary files: `config.py`, model config/core modules, and `train/config/{schema,resolution,artifacts}.py` plus `train/cli.py` under `src/speculators/`.

- [ ] Add serializable typed configuration for backbone, correction, confidence, and objective. Reuse the existing class-registry conventions rather than introducing another discovery system without need.
- [ ] Add one modular model/config registration that can assemble supported components from configuration. Preserve existing registered model/config classes as compatibility entry points delegating to shared assembly.
- [ ] Define existing algorithm names as presets with explicit component settings. Preserve current defaults; distinguish omitted values from explicit user overrides.
- [ ] Add YAML-first component selection for mixed recipes. Preserve existing CLI flags and translate them to component configuration at the boundary.
- [ ] Specify precedence and conflicts among preset selection, explicit component overrides, `--draft-config`, and `--from-pretrained`. Reject conflicting architecture changes when resuming weights or training state.
- [ ] Separate architecture configuration from training-only objective hyperparameters. Save every setting necessary for inference in model config and resolved training settings in the run configuration.
- [ ] Preserve `SpeculatorsConfig.algorithm` for legacy serving exports. Give native mixed recipes an honest architecture identity and explicit proposal requirements.
- [ ] Ensure generic `SpeculatorModel.from_pretrained()` dispatches native and legacy configurations without ambiguity and external-checkpoint detection still works.
- [ ] Update `get_trainer_kwargs()` or replace its internal implementation with objective-derived kwargs while preserving its public compatibility until callers migrate.
- [ ] Update optimizer classification where renamed correction parameters affect AdamW/Muon routing. Verify every trainable parameter belongs to exactly one intended group.
- [ ] Write resolved component selections, versions, objective settings, and source checkpoint identities into run artifacts. Preserve training command, code patch, and existing provenance behavior.
- [ ] Add one example configuration for DFlash2 backbone plus Markov correction and confidence prediction, using the matching DSpark-style objective. Reject feature-incompatible combinations with actionable errors.

Completion: the mixed recipe constructs and trains through configuration without introducing a subclass for that combination.

## 9. Phase G: serving and later backbone families

- [ ] Audit each currently supported serving loader's expected parameter keys, architecture names, config fields, and proposal semantics. Record supported versions before implementing runtime-specific export.
- [ ] Keep legacy serving exports numerically equivalent for existing recipes. Extend current attention parity tests, but do not treat them as proof that an exported checkpoint loads in a real serving engine.
- [ ] Add an actual load-and-draft integration test against the supported serving implementation for each export path being claimed.
- [ ] For the first mixed recipe, implement matching serving composition or explicitly mark it training-only until that support exists. Correct serialized weights alone are insufficient for sequential Markov conditioning or iterative XPress decoding.
- [ ] Validate masks, cache updates, predecessor alignment, candidate scoring, and refinement iteration behavior in the runtime that will execute them.
- [ ] Run a small serving smoke test and record acceptance/throughput measurements only when actually measured. Include provenance with any published model or evaluation artifacts.
- [ ] Always pass `--provenance-dir` when using `scripts/launch_vllm.py`; retain associated training/eval commands, launch commands, checkpoint hashes, and patches.
- [ ] After the block-family design is validated, inventory EAGLE3/P-EAGLE autoregressive state and MTP step-specific semantics. Extend preparation, output alignment, and inference-state contracts only for concrete supported combinations.
- [ ] Implement at least one cross-family combination before claiming backbones are interchangeable across all speculative-decoding algorithms. Keep unsupported combinations rejected rather than hidden behind shape-compatible adapters.

Completion of the initial scope requires preserved serving support for existing recipes and an explicit support status for the new mixture. Cross-family migration is a follow-on milestone, not an implicit promise of the first release.

## 10. Validation plan

Add tests that verify observable behavior and compatibility rather than mirroring class construction. Keep verifier fixtures local or explicitly mocked; network access is not required for unit tests.

| Area                        | Required evidence                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Existing recipes            | Fixed-weight, fixed-anchor output/loss/gradient equivalence to the pre-refactor implementation                        |
| Slot and document alignment | Both anchor conventions, nontrivial position IDs, packed documents, padding, sparse/all-zero masks, and short inputs  |
| Backbone variants           | Correct layer types, convolution initialization, full/sliding attention, and target-feature dimensions                |
| Corrections                 | Identity, all Markov variants, confidence feature modes, selector policies, XPress training and iterative evaluation  |
| New composition             | DFlash2 backbone plus Markov correction with finite loss, expected gradients, valid config, and checkpoint round trip |
| Legacy compatibility        | Real legacy key fixtures, strict learned-weight validation, conversion, and matching serving export                   |
| Native serialization        | Config dispatch, safetensors sharding, dtype, omitted verifier weights, and collision/version errors                  |
| Training infrastructure     | Single-device setup, DDP/FSDP wrapping, gradient checkpointing, mixed precision, and frozen replication               |
| Resume                      | Correct model/optimizer assignments and next-step equivalence for each supported format and optimizer combination     |
| Execution performance       | Eager/compiled parity, dynamic anchor behavior, and no unexplained extra backbone calls or full-logit materialization |
| Serving                     | Actual checkpoint load and draft execution for each claimed runtime format                                            |

Start with the existing focused suite:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/unit/models/test_dflash_utils.py \
  tests/unit/models/test_dflash_targets.py \
  tests/unit/models/test_dflash2_model_definitions.py \
  tests/unit/convert/test_dflash_converter.py
```

As each phase lands, add its new regression tests and run the relevant existing suites:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/unit/models tests/unit/convert tests/unit/train \
  tests/unit/test_model.py tests/unit/test_config.py
```

Run the appropriate GPU integration cases from `tests/integration/models/`, single-device and multi-GPU training/resume smoke tests, and actual serving tests in an environment with their dependencies. A skipped GPU test is not GPU validation. Keep original model families in regression coverage while migrating common trainer interfaces.

After the implementation changes pass their targeted checks, run repository quality checks (`make quality`) using the development environment. Install `[dev]` extras and `pre-commit install` once per clone when preparing implementation contributions. Report commands, pass/fail status, environment limitations, and measured results accurately.

## 11. Suggested delivery order

1. **Reference fixtures and interface contracts:** Phases A and B; document layouts and compatibility promises.
2. **Usable backbone extraction:** Phase C plus the model-weight mapping and legacy export portions of Phase D required to keep the branch usable. Migrate all existing callers together.
3. **Complete checkpoint/resume compatibility:** Finish Phase D, including distributed loading and optimizer-state handling before claiming resume support.
4. **Corrections and objectives:** Phase E, keeping named recipes numerically unchanged.
5. **Configurable composition:** Phase F and the first mixed-recipe example/test.
6. **Serving and release documentation:** Initial-scope Phase G work, explicit support matrix, examples, and migration instructions.
7. **Additional backbone families:** Subsequent concrete cross-family experiments and any necessary interface extensions.

Keep each delivery step reviewable and preserve working named recipes at integration points. Use `jj` for local change inspection and organization. When preparing PRs, follow the repository's Purpose/Tests guidance and leave the human-review checkbox unchecked.

## 12. Definition of done

- [ ] Existing DFlash, DFlash2, DSpark, and XPress behavior is preserved by numerical and training regression checks.
- [ ] Backbones, correction, and confidence modules have single ownership and explicit contracts.
- [ ] The first mixed recipe works through configuration without another algorithm subclass.
- [ ] Legacy weights and native modular weights load through public APIs and checkpointer paths.
- [ ] Existing supported serving recipes retain a validated export path; new mixtures have an explicit serving support status.
- [ ] Supported full-resume paths restore the intended optimizer state; unsupported legacy states fail explicitly with a documented weights-only alternative.
- [ ] Configuration, provenance, examples, migration notes, and developer guidance describe the implemented support rather than the intended future scope.
- [ ] Required checks pass, or remaining environment-dependent checks and their effect on release readiness are recorded explicitly.
