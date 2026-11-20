# Review output format

A review is one `review.json` file. `scripts/review_payload.py` checks it and turns it into a GitHub review.

```json
{
  "version": 1,
  "pr": 1234,
  "summary": "",
  "findings": [
    {"confidence": 90, "file": "src/speculators/x.py", "line": 56, "body": "This recomputes the `argmax` the fused pass already returned; use `out.pred_ids` instead."},
    {"confidence": 85, "file": "src/speculators/x.py", "start_line": 40, "line": 44, "body": "A NaN validation loss never compares lower, so `checkpoint_best` stays on the first epoch. Skip non-finite losses here.", "detail": "`nan < best` is `False` for every `best`, so the comparison on line 42 never passes again. `if math.isfinite(val_loss) and val_loss < best:` fixes it."},
    {"confidence": 80, "body": "The description has no Tests section, which [AGENTS.md › Pull requests](https://github.com/vllm-project/speculators/blob/{base.sha}/AGENTS.md#pull-requests) asks for."}
  ]
}
```

- `pr` is the number of the PR in the context directory; a review for any other PR is refused.
- `summary` is markdown and usually empty: it holds only what no finding can (see the rubric). It is posted only when at least one finding survives.
- `body` is markdown: the comment as the author first reads it. `detail` is optional markdown, posted under the body in a collapsed section (see the rubric for what goes where).
- A finding about the PR as a whole (description, contribution rules, splitting) has no `file` or `line`. It goes into the review body.
- `confidence` is the 0–100 score from the rubric. Findings below 80 are dropped, PR-wide ones included.
- `file` is the path as it appears after `+++ b/` in the diff.
- `line` is the line number in the new version of the file, counted from the `+N` of its `@@ -M,K +N,L @@` hunk header over context (` `) and added (`+`) lines only. Removed (`-`) lines have no new-file number and can't carry a comment.
- `start_line` is optional and marks a range ending at `line`. Both lines must be in the same hunk.
- A finding whose lines aren't commentable is not lost: it is appended to the review body with its location.

Before you finish, check the draft and fix what it reports (`scripts/` is next to this `references/` directory):

```bash
python3 scripts/review_payload.py review.json <context_dir> --check
```

It validates the format and measures each body that would be posted: at most 50 words, and no sentence over 30. A body over that is carrying its reasoning, which belongs in `detail`.

The review is always posted as a comment. It never approves and never requests changes; a maintainer makes that call.
