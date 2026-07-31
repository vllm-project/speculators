# Draft-OPD: On-Policy Distillation for Speculative Draft Models

Implementation of Draft-OPD ([arXiv:2605.29343](https://arxiv.org/abs/2605.29343)) in the speculators training framework.

## Overview

Draft-OPD is a post-SFT training stage that closes the offline-to-inference distribution mismatch in speculative draft models. SFT trains on fixed target-generated trajectories, so the drafter never sees its own prediction errors. Draft-OPD fixes this with an acceptance-aware loss: forward KL on accepted positions (where draft matches target) and reverse KL with position decay on rejected positions (where draft diverges).

Key result from the paper: 5x+ lossless speedup on Qwen3 thinking models, +13% over DFlash and +23% over EAGLE-3 at matched FLOPs.

## Architecture

No vLLM changes are required. vLLM serves as a hidden-state extraction service via its existing `extract_hidden_states` mode. All OPD logic lives on the speculators side.

### Training loop (per prompt)

```
1. Generate verified sequence
   Send prompt to vLLM → target generates response → extract hidden states
   Result: {hidden_states, token_ids} for the verified sequence

2. Draft model forward (with grad)
   Run _backbone_forward on verified hidden states → draft logits + target logits
   Determine draft proposals via argmax(draft_logits)

3. Build accept/reject mask
   Compare argmax(draft) vs argmax(target) within each block
   Sequential acceptance via cumprod (first mismatch kills rest of block)

4. Replay rejected blocks via vLLM
   For each block with rejections:
     Construct [context_up_to_anchor + draft_proposals]
     Send to vLLM extract_hidden_states → get target hidden states
       conditioned on draft prefix (not verified prefix)
     Compute correct target logits via shared LM head
   Accepted blocks use verified hidden states (already correct)

5. OPD loss → backprop → optimizer step
   L_acc = mean forward KL on accepted positions
   L_rej = decay-weighted mean reverse KL on rejected positions
   loss = (λ_acc · L_acc + λ_rej · L_rej) / (λ_acc + λ_rej)
```

### Why replay is needed

During speculative decoding, the target model verifies draft-proposed tokens. When a draft token is rejected, the target's hidden states at subsequent positions are conditioned on the *verified* prefix (with a resampled bonus token), not the draft's rejected token. For the OPD loss to correctly penalize rejected positions, we need the target's distribution conditioned on the draft prefix — which requires re-running the target model on the draft-proposed tokens.

For accepted positions and the first rejected position per block, the verified hidden states are already correct (same prefix). The extra vLLM call is only needed for 2nd+ rejected positions in each block.

### Accept mask construction

At temperature=0 (greedy, the paper's primary setting), `argmax(draft) == argmax(target)` exactly matches speculative verification. For temperature>0, this is approximate.

```python
match = (draft_preds == target_preds).view(num_anchors, block_size)
match[:, 0] = True  # anchor slot (sample_from_anchor=False)
accept_mask = match.cumprod(dim=-1)  # sequential acceptance
```

## Files

| File | Description |
|------|-------------|
| `src/speculators/models/opd_loss.py` | OPD loss module: `build_accept_mask`, `opd_loss`, `compute_opd_metrics` |
| `src/speculators/models/dflash/core.py` | Modified `forward()` with `training_stage="opd"` support |
| `scripts/train_opd.py` | OPD training controller (online loop with vLLM) |
| `scripts/prepare_opd_prompts.py` | Prepare the 16K prompt pool from HuggingFace datasets |
| `tests/unit/models/test_opd_loss.py` | Unit tests for the loss module |
| `data/opd_prompts_raw.jsonl` | Raw OPD prompts from [Draft-OPD repo](https://github.com/bingyang-lei/Draft-OPD) (16K: 4K AoPS + 5K code + 5K math + 2K GSM8K) |

## Usage

### Step 1: Prepare data

The paper's 16K prompt pool is already downloaded at `data/opd_prompts_raw.jsonl`. To regenerate responses with the target model:

```bash
# Uses the response_regeneration pipeline to generate responses via vLLM
cd /home/shanjiaz/speculators
./scripts/response_regeneration/run_all.sh \
    --model Qwen/Qwen3-8B \
    --dataset opd \
    --max-tokens 4096
```

This starts a vLLM server, sends all 16K prompts, generates responses, and saves the output as jsonl. The `opd` dataset config is registered in `src/speculators/data_generation/configs.py`.

### Step 2: Generate hidden states

After regeneration, generate hidden states for the training data using the standard pipeline:

```bash
# Launch vLLM with extract_hidden_states
python scripts/launch_vllm.py \
    --model Qwen/Qwen3-8B \
    --speculative-method extract_hidden_states \
    --draft-model /path/to/sft_checkpoint

# Generate hidden states
python scripts/data_generation_offline.py \
    --data-path /path/to/regenerated_data \
    --vllm-endpoint http://localhost:8000
```

### Step 3: Train with OPD

#### Option A: Online training (generates verified sequences on the fly)

```bash
python scripts/train_opd.py \
    --vllm-base-url http://localhost:8000 \
    --model Qwen/Qwen3-8B \
    --prompt-data data/opd_prompts.jsonl \
    --from-pretrained /path/to/sft_checkpoint \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --epochs 8 \
    --lr 3e-4 \
    --gamma-opd 0.8 \
    --lambda-acc 1.0 \
    --lambda-rej 1.0 \
    --max-response-tokens 4096 \
    --save-path /path/to/output
```

Note: `--prompt-data` expects a jsonl file with `{"input_ids": [...]}` per line (tokenized prompts). Use `scripts/prepare_opd_prompts.py` to generate this from the raw prompts.

#### Option B: Use existing SFT trainer with OPD loss

If you have pre-generated hidden states, you can use the standard training script with the OPD training stage flag:

```bash
python scripts/train.py \
    --speculator-type dflash \
    --from-pretrained /path/to/sft_checkpoint \
    --data-path /path/to/hidden_states \
    --training-stage opd \
    --gamma-opd 0.8 \
    --lambda-acc 1.0 \
    --lambda-rej 1.0 \
    ...
```

This uses the verified sequence hidden states for all positions (no vLLM replay for rejected blocks). The accept/reject mask is reconstructed via argmax comparison. This is approximate at positions after the first rejection per block but captures the primary signal.

## Paper hyperparameters

From Draft-OPD (arXiv:2605.29343), Appendix A:

| Parameter | Value |
|-----------|-------|
| SFT epochs before OPD | 6 |
| OPD epochs | 8 |
| Position decay γ | 0.8 |
| λ_acc, λ_rej | 1.0, 1.0 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Schedule | Cosine with 5% warmup |
| Max response length (thinking) | 4096 |
| Max response length (non-thinking) | 2048 |
| Block size | 16 |
| Draft model layers | 5 (4B/8B), 8 (30B-A3B) |
| Prompt pool | 16K (2K GSM8K + 5K MATH + 4K AoPS + 5K CodeAlpaca) |
| Training framework | verl (OPD stage) |

## Loss formulation

**Accepted positions — Forward KL (mode-covering):**
```
L_acc = (1/|I_acc|) Σ_{accepted} D_KL(p_target || q_draft)
```

**Rejected positions — Reverse KL (mode-seeking) with position decay:**
```
L_rej = (1/Z) Σ_{rejected} γ^k · D_KL(q_draft || p_target)
```
where `Z = Σ_{rejected} γ^k` and `k` is the 0-indexed position within the block.

**Combined:**
```
L = (λ_acc · L_acc + λ_rej · L_rej) / (λ_acc + λ_rej)
```

Forward KL on accepted tokens makes the draft cover the target's full distribution at states where the draft is already close. Reverse KL on rejected tokens penalizes draft modes the target doesn't support. Position decay down-weights later rejected positions since they're increasingly off-distribution due to earlier errors.

## References

- Draft-OPD paper: [arXiv:2605.29343](https://arxiv.org/abs/2605.29343)
- Official repo: [github.com/bingyang-lei/Draft-OPD](https://github.com/bingyang-lei/Draft-OPD)
- Training data: [Draft-OPD/data](https://github.com/bingyang-lei/Draft-OPD/tree/main/data)
