# Review output format

A review is one `review.json` file. `scripts/review_payload.py` checks it and turns it into a GitHub review.

```json
{
  "version": 1,
  "pr": 1234,
  "summary": "Design looks good; two gaps below.",
  "findings": [
    {"confidence": 90, "file": "src/speculators/models/peagle/metrics.py", "line": 56, "body": "This still runs its own `argmax`; pass `with_ids=True` and use `out.pred_ids`, as DFlash does."},
    {"confidence": 85, "file": "src/speculators/train/trainer.py", "start_line": 40, "line": 44, "body": "A NaN validation loss never compares lower, so `checkpoint_best` stays on the first epoch. Skip non-finite losses here."},
    {"confidence": 80, "body": "The description has no Tests section, which AGENTS.md asks for."}
  ]
}
```

- `pr` is the number of the PR in the context directory; a review for any other PR is refused.
- `summary` is markdown: your verdict only, not a description of the PR (see the rubric). It is posted only when at least one finding survives.
- A finding about the PR as a whole (description, contribution rules, splitting) has no `file` or `line`. It goes into the review body.
- `confidence` is the 0–100 score from the rubric. Findings below 80 are dropped, PR-wide ones included.
- `file` is the path as it appears after `+++ b/` in the diff.
- `line` is the line number in the new version of the file, counted from the `+N` of its `@@ -M,K +N,L @@` hunk header over context (` `) and added (`+`) lines only. Removed (`-`) lines have no new-file number and can't carry a comment.
- `start_line` is optional and marks a range ending at `line`. Both lines must be in the same hunk.
- A finding whose lines aren't commentable is not lost: it is appended to the review body with its location.

The review is always posted as a comment. It never approves and never requests changes; a maintainer makes that call.
