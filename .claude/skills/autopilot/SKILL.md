---
name: autopilot
description: Full autonomous loop — discover new speculative decoding papers, then run oneshot-paper on each promising one. Completely hands-off.
---

# Autopilot: Discover + Implement Pipeline

You are running the full autonomous pipeline: discover new speculative decoding methods, then implement and smoke-train each promising one. No human interaction needed.

## Step 1: Discovery

Follow `.claude/skills/discover.md` fully (search arxiv, HF, GitHub, deduplicate, classify). But instead of asking the user which to pursue, apply this filter automatically:

### Auto-pursue criteria

The user may pass `--days N` to override the default 60-day recency window (e.g. `/autopilot --days 120`).

Only pursue items classified as **new-method** or **variant** that meet ALL of:
1. Not in `.claude/agent_state/blacklist.txt` (check paper URL, arxiv ID, and method name against entries, case-insensitive)
2. Published within the last N days (default: 60)
3. Has a clear algorithmic description (not just benchmark results)
4. Is about speculative decoding for autoregressive LLMs (not diffusion, not vision, not audio)
5. Is not a trivial modification (e.g., just changing hyperparameters of an existing method)

Skip items classified as **new-base-model** or **irrelevant**.

If no items pass the filter, report:
```
## Autopilot Scan Complete
No new methods found worth implementing. Next scan in ~8 hours.
Scanned: N papers, M models, K repos
```
And stop.

## Step 2: For Each Qualifying Item

For each paper/model that passes the filter, run the full `/oneshot-paper` pipeline:

1. Read the paper URL
2. Invoke the oneshot-paper skill instructions (analyze → implement → train)
3. Collect the results

**Important**: Run items sequentially, not in parallel. Each implementation creates a separate git branch (`feat/<algo_name>`), so they won't conflict, but training requires GPU resources.

## Step 3: Summary Report

After processing all items (or if none found), present a single summary:

```markdown
## Autopilot Report

### Discovery
- Sources scanned: arxiv, HuggingFace, GitHub
- New items found: N
- Passed filter: M
- Previously seen (skipped): K

### Implementations Attempted

#### 1. <algo_name_1>
- Paper: <title> (<url>)
- Branch: feat/<algo_name_1>
- Implementation: PASS/FAIL
- Training: PASS/FAIL (initial loss X.XX → final loss X.XX)

#### 2. <algo_name_2>
...

### Action Items
- [List branches ready for review]
- [List failed implementations that need manual attention]
- [List any papers that looked promising but were skipped and why]
```

## Cron Integration

This skill is designed to be called by the durable cron job. Update the cron to use this instead of bare `/discover`:

The cron at `17 */8 * * *` should invoke `/autopilot` for fully hands-off operation, or `/discover` if you only want monitoring without auto-implementation.
