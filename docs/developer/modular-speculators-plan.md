# Modular speculative decoding implementation plan

## Scope

The modular model combines a DFlash or DFlash2 backbone with an identity, Markov, candidate-selector, or XPress correction. Confidence prediction is available with identity and Markov correction. The DFlash2 plus Markov example exercises a mixed recipe through the shared model. Modular serving and composition with EAGLE3, P-EAGLE, or MTP are outside this milestone.

## Component boundaries

- The model wrapper assembles components and owns vocabulary mapping, vocabulary projection, and verifier-weight reconstruction.
- Input preparation selects anchors and builds the block layout, token embeddings, masks, and rotary positions. The backbone consumes those inputs and returns normalized hidden states.
- A correction uses backbone features and caller-supplied predecessor tokens to modify logits or score candidates. The training objective chooses the training context, builds verifier targets, and computes loss and metrics.
- Each learned module has one registration path. The training wrapper exposes the decoder layers to FSDP through a shared accessor.

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

## Checkpoints

DFlash, DFlash2, DSpark, and XPress use flat checkpoint names. A shared prefix mapping translates between those names and the registered module paths. Modular checkpoints use the hierarchical module paths directly.

Full training resume requires the same Speculators version that wrote the checkpoint. Use `--from-pretrained` to initialize a new run from compatible weights.

## Validation and remaining work

The numerical contract tests compare the named algorithms against pre-refactor revision `988ed16b1f91315942049da3093d88dba70c45d9` under both anchor conventions. Unit tests also exercise mixed-recipe training, sharded checkpoint round trips, and the next optimizer update after same-version resume.

- Verify a multi-GPU FSDP checkpoint save and resume with a modular recipe.
- Verify that vLLM loads and drafts from a named model's flat checkpoint. Use `--provenance-dir` when launching vLLM through `scripts/launch_vllm.py`.

See [Modular block models](../user_guide/algorithms/modular.md) for recipe configuration and training commands.
