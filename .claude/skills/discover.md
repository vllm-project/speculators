---
name: discover
description: Scan arxiv, HuggingFace, and GitHub for new speculative decoding papers and models. Deduplicates against a local tracking log and presents a summary report.
---

# Discovery Monitor for New Speculative Decoding Methods

You are scanning for new speculative decoding research and models. Your goal is to find papers/models describing **new draft model architectures or training methods** for speculative decoding in LLMs.

## Step 1: Load State

Read the tracking file at `.claude/agent_state/discovery_log.json`. If it doesn't exist, start with an empty state:
```json
{"seen_papers": {}, "seen_models": {}, "seen_repos": {}, "last_scan": null}
```

## Step 2: Search Sources

Run these searches and collect results:

### arxiv
Use `WebSearch` with these queries (use current year):
1. `"speculative decoding" new method arxiv YYYY`
2. `"draft model" "language model" speculation arxiv YYYY`
3. `"eagle" OR "medusa" OR "hydra" OR "kangaroo" speculative decoding arxiv YYYY`
4. `"multi-token prediction" draft speculation arxiv YYYY`

Extract arxiv IDs (e.g., `2506.12345`) from results.

### HuggingFace
Use `WebFetch` on:
- `https://huggingface.co/api/models?search=speculative+decoding&sort=lastModified&direction=-1&limit=20`
- `https://huggingface.co/api/models?search=eagle+draft&sort=lastModified&direction=-1&limit=10`
- `https://huggingface.co/api/models?search=medusa+draft&sort=lastModified&direction=-1&limit=10`

### GitHub
Use bash:
```bash
gh api "search/repositories?q=speculative+decoding+language:python+pushed:>$(date -d '30 days ago' +%Y-%m-%d)&sort=updated&per_page=10"
```

## Step 3: Deduplicate

Compare all found items against `seen_papers`, `seen_models`, and `seen_repos` in the tracking file. Only keep items not previously seen.

## Step 4: Classify Each New Item

For each new item, classify it as one of:
- **new-method**: Describes a fundamentally new speculative decoding architecture (new attention pattern, new loss, new hidden state fusion technique) not currently implemented in this codebase
- **variant**: A variant or improvement of an existing method (eagle3, dflash, dspark, peagle, mtp) that could be added by extending an existing model class
- **new-base-model**: An existing speculator type (eagle3, etc.) applied to a new base model (a new LLM architecture)
- **irrelevant**: Uses speculative decoding but doesn't introduce a new method worth implementing

The currently implemented methods in this codebase are:
- **eagle3**: EAGLE-3 with aux hidden states, FC projection, TTT-step autoregressive drafting
- **dflash**: DFlash with mask tokens and parallel drafting
- **dspark**: DSpark (extends DFlash with Markov logit bias + confidence head)
- **peagle**: PEagle (extends Eagle3 with parallel multi-token prediction + COD sampling)
- **mtp**: Multi-Token Prediction (DeepSeek-style)

## Step 5: Update State

Write the updated tracking file back to `.claude/agent_state/discovery_log.json`, adding all newly seen items with today's date and status "new".

## Step 6: Present Report

Present a summary table to the user:

```
## Speculative Decoding Discovery Report

### New Findings
| # | Type | Title | Source | URL | Classification |
|---|------|-------|--------|-----|----------------|
| 1 | paper | ... | arxiv | ... | new-method |
| 2 | model | ... | HF | ... | variant |

### Recommendations
- [List items worth pursuing, especially "new-method" and "variant" classifications]
- [For each, note which existing implementation is closest]

### Previously Seen (skipped)
- N papers, M models already tracked
```

Ask the user which items they'd like to pursue. For selected items, suggest running `/analyze-paper <url>`.
