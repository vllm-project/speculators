# DSpine

DSpine extends [DFlash](dflash.md) with causal conditioning throughout the parallel draft backbone. It is based on [Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding](https://arxiv.org/abs/2609.36173).

## How It Works

Every draft layer first updates the complete anchor-plus-mask block in parallel. It then projects each position into a compact transfer space derived from the frozen verifier LM head. A gated adjacent injection writes the feature predicted at position `t - 1` into position `t`. The readout and residual-write matrices are layer-specific; the message and gate transforms are shared across layers. The anchor remains unchanged, and no operation crosses a draft-block boundary.

The anchor is already a known token: its outgoing message uses its fixed transfer embedding at every layer, independently of the correct-predecessor curriculum. Only masked predecessors use predicted features when curriculum substitution is disabled. This rule must match in training and serving.

Fresh construction through the training CLI enables `known_anchor_injection=true` and saves it in the checkpoint. Loading with `--from-pretrained` honors the saved configuration instead; checkpoints without this field retain the legacy predicted-anchor behavior in both training and serving. Opting an old checkpoint into the corrected mode requires an explicit configuration copy and changes its computation; it is an ablation, not an automatically improved checkpoint.

DSpine uses causal attention inside every synthetic block. Information received from an earlier position at one layer can therefore influence the feature sent to the next position at the following layer, while all positions within a layer remain parallel.

The transfer embeddings are built once from normalized verifier output embeddings: the implementation centers them, computes the leading PCA directions, whitens the projected rows, and normalizes them. Predicted features and decided tokens then share one space.

## Last-Write Refinement and Transition Cache

The backbone LM head first produces a unary top-K candidate set per position. For a decided predecessor token, DSpine replaces the final layer's predicted predecessor feature with that token's fixed transfer embedding and repeats only the residual write. `build_transition_cache` vectorizes this operation for every adjacent pair of top-K candidate sets and returns a `[blocks, positions, K, K]` score tensor. A serving runtime can select through this cache without sequential matrix multiplications.

Speculators implements the model, training objective, checkpoint format, and transition-cache primitive. Serving requires a runtime with DSpine checkpoint loading and proposal selection support; it is not interchangeable with the existing DFlash or DFlash2 runtime path.

## Training Objective

Training follows the paper:

1. Unary backbone logits receive a weighted mixture of hard-token cross-entropy and L1 distance to the verifier distribution.
2. Last-write refinement receives the same objective over the unary top-K set. Cross-entropy is used only when that set contains the target token; L1 is used when the set contains more than the configured minimum verifier mass.
3. Every layer's post-injection feature receives cosine supervision toward the correct token's transfer embedding.

Hard-token CE, top-K label membership, and embedding supervision use the actual token IDs in the training sequence. The verifier distribution is the soft target for the L1 term; its argmax must not silently replace sequence labels.

The embedding-loss multiplier warms linearly and is scaled by a detached unary cross-entropy. During the first part of training, a block-level curriculum can substitute correct-token predecessor features in the shallow layers; it then decays to zero.

## Paper Defaults

| Parameter                        |  Default |
| -------------------------------- | -------: |
| Draft layers                     |        5 |
| Block size                       |       16 |
| Transfer dimension               |     1024 |
| Message dimension                |      512 |
| Transition candidates            |       16 |
| Position decay gamma             |        7 |
| CE mixture weight                |      0.1 |
| Embedding beta                   |      0.5 |
| Embedding warmup                 |      146 |
| Curriculum probability           |      0.5 |
| Curriculum hold / end            | 1/6, 1/3 |
| Shallow curriculum layers        |        3 |
| Minimum top-K target mass for L1 |     1e-4 |

DSpine requires the full verifier vocabulary. The fixed transfer embeddings and shared LM head make reduced-vocabulary checkpoints invalid.

For the paper's Qwen3-8B shape, use target feature layers `1 9 17 25 33`, a maximum sequence length of 3072, 512 anchors per sequence, AdamW with no weight decay and peak learning rate `6e-4`, 146 warmup steps, cosine decay, and six epochs. The source data should be regenerated by the pinned verifier with thinking disabled.

```bash
torchrun --standalone --nproc_per_node=8 -m speculators.train \
  --speculator-type dspine \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./data/qwen3-8b-regenerated \
  --save-path ./output/qwen3-8b-dspine \
  --target-layer-ids 1 9 17 25 33 \
  --num-layers 5 \
  --block-size 16 \
  --max-anchors 512 \
  --total-seq-len 3072 \
  --optimizer adamw \
  --weight-decay 0 \
  --lr 6e-4 \
  --scheduler-type cosine \
  --scheduler-warmup-steps 146 \
  --epochs 6
```

Before a full run, use `--dry-run`, validate checkpoint construction, and run a one-batch training/save/reload gate with the exact verifier and hidden-state extractor revisions intended for the experiment. `examples/train/dspine_qwen3_8b_dry_run.sh` builds an initialized checkpoint and checks its known-anchor and 15-proposal settings without starting training.

## See Also

- [DFlash](dflash.md) -- The underlying block-parallel draft backbone
- [DFlash2](dflash2.md) -- Local convolution plus post-backbone selection
- [DSpark](dspark.md) -- Sequential Markov correction after the backbone
- [Train a Speculator](../tutorials/train.md) -- End-to-end training workflow
