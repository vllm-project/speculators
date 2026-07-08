---
name: analyze-paper
description: Read a speculative decoding paper and produce a structured implementation spec for the speculators and vLLM codebases.
---

# Paper Analysis for Speculative Decoding Methods

You are reading a research paper describing a speculative decoding method and producing a structured implementation specification. The user provides a URL (arxiv, PDF path, blog post, or HuggingFace model card).

## Step 1: Acquire the Paper

Based on the URL type:
- **arxiv abstract** (`arxiv.org/abs/XXXX.XXXXX`): Use `WebFetch` on both the abstract page AND the HTML version (`arxiv.org/html/XXXX.XXXXX`). The HTML version has the full paper content.
- **arxiv PDF** (`arxiv.org/pdf/XXXX.XXXXX`): Fetch the abstract page instead for text content, then try the HTML version.
- **Local PDF**: Use `Read` on the file path.
- **HuggingFace model**: Use `WebFetch` on the model page for the README/model card. Also fetch the `config.json` from the model repo.
- **Blog post / other URL**: Use `WebFetch` directly.

If the paper references a code repository, also fetch that for implementation details.

## Step 2: Understand the Existing Codebase

Before analyzing the paper, understand what already exists. Read these files:
- `src/speculators/models/__init__.py` — see all implemented model types
- `src/speculators/models/eagle3/core.py` — the primary reference implementation
- `src/speculators/models/eagle3/config.py` — the reference config pattern
- `src/speculators/models/dflash/core.py` — alternative architecture pattern
- `docs/developer/add_algorithm.md` — the official guide for adding new algorithms

## Step 3: Extract the Method

Systematically identify:

### Architecture
- What type of draft model is it? (autoregressive like Eagle3, parallel like DFlash, tree-based, hybrid)
- How are verifier hidden states used? (concatenation, addition, attention, gating)
- How many auxiliary hidden states from the verifier are needed? Which layers?
- What is the decoder layer structure? (standard transformer, modified attention, custom layers)
- What is the prediction head? (single lm_head, multiple heads like Medusa, confidence head)
- Does it use vocabulary mapping (draft vocab != target vocab)?

### Training
- What loss function? (cross-entropy, KL divergence, reverse KL, custom)
- Online or offline data generation?
- Does it use teacher forcing, off-policy tokens, or autoregressive TTT steps?
- Any special data preprocessing needed?
- Any special optimizer or learning rate schedule?

### Inference
- How does token proposal work? (greedy, tree-based, parallel)
- How does verification work? (standard spec decode rejection, custom acceptance criterion)
- Does it map to an existing vLLM proposer? (EagleProposer for autoregressive, DFlashProposer for parallel, MedusaProposer for multi-head)

### Key Innovation
- What is the core difference from existing methods?
- Is this an extension of an existing method or a completely new approach?

## Step 4: Determine Implementation Strategy

Decide:
1. **Closest existing implementation**: Which model in `src/speculators/models/` is most similar?
2. **Inheritance or standalone**: Can this extend an existing model (like DSpark extends DFlash) or does it need to be standalone?
3. **vLLM pathway**: Does the inference method map to an existing proposer (eagle, dflash, medusa) or need a new one?

## Step 5: Produce the Implementation Spec

Write a structured spec to `.claude/agent_state/specs/<algo_name>.md` with this format:

```markdown
# Implementation Spec: <Algorithm Name>

## Paper
- Title: ...
- URL: ...
- Key authors: ...

## Summary
<2-3 sentences describing what the method does and its key innovation>

## Classification
- Type: new-method | variant-of-<parent>
- Closest existing: eagle3 | dflash | mtp | none
- Inheritance: standalone | extends <ParentModel>

## Speculators Implementation

### Config (`src/speculators/models/<name>/config.py`)
- speculators_model_type: "<name>"
- Fields:
  - field_name: type = default  # description
  - ...
- Parent class: SpeculatorModelConfig | Eagle3SpeculatorConfig | DFlashSpeculatorConfig

### Model (`src/speculators/models/<name>/core.py`)
- Parent class: SpeculatorModel | Eagle3DraftModel | DFlashDraftModel
- Architecture:
  - Layers: [describe decoder layer structure]
  - FC/projection: [describe input projection]
  - Norms: [describe normalization layers]
  - Heads: [describe prediction heads]
- target_layer_ids: [which verifier layers are needed]
- forward() pseudocode:
  ```
  1. Project hidden states through FC
  2. For each TTT step:
     a. Embed input tokens
     b. Concatenate embeddings + hidden states
     c. Run through decoder layers
     d. Compute logits via lm_head
     e. Compute loss against verifier targets
     f. Select next input tokens (argmax)
  3. Return (draft_tokens, loss, metrics)
  ```
- from_training_args() fields needed: [list kwargs consumed]
- get_trainer_kwargs() fields: [list training kwargs]

### Data preprocessing
- Needs custom shift_batch? yes/no
- If yes, describe the preprocessing

### CLI arguments for train.py
- --arg-name: type, default, description
- ...

## vLLM Implementation

### Config translation (`algos.py`)
- Maps to existing method: eagle3 | dflash | new
- Architecture name: <ArchNameForCausalLM>
- Config fields to set: [list]

### Model (`vllm/model_executor/models/`)
- Reuses existing model class: yes (<class>) | no (needs new file)
- If new, describe the nn.Module structure

### Proposer / Speculator
- Reuses existing: EagleProposer | DFlashProposer | MedusaProposer
- Or needs new proposer: [describe]

### Registry entries
- _SPECULATIVE_DECODING_MODELS entries needed
- SpeculativeMethod additions (if any)

## Training
- Recommended verifier for smoke test: [model name]
- Recommended hyperparameters:
  - lr: ...
  - epochs: ...
  - ttt_steps (if applicable): ...
  - loss_fn: ...
- Expected behavior: [what should training loss look like]

## Risks and Unknowns
- [List anything unclear from the paper]
- [List any assumptions made]
```

## Step 6: Present to User

Show the user:
1. A brief summary of the paper and method
2. The classification and implementation strategy
3. Any risks or unknowns that need human judgment
4. Ask for approval to proceed with `/implement-speculator <algo_name>`
