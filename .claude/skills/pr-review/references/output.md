# Review output format

A review is one `review.json` file. `scripts/review_payload.py` turns it into a GitHub review; [review.schema.json](review.schema.json) is the same format as a JSON Schema.

```json
{
  "version": 1,
  "pr": 1234,
  "summary": "1–3 sentences on the overall design assessment, plus PR-wide findings as a bulleted list.",
  "findings": [
    {"confidence": 90, "file": "src/speculators/x.py", "line": 42, "body": "The problem in one sentence, then the fix."},
    {"confidence": 85, "file": "src/speculators/x.py", "start_line": 40, "line": 44, "body": "A finding about a range of lines."}
  ]
}
```

- `summary` is markdown. Findings about the PR as a whole (description, contribution rules, splitting) go here, not in `findings`.
- `confidence` is the 0–100 score from the rubric. Findings below 80 are dropped.
- `file` is the path as it appears after `+++ b/` in the diff.
- `line` is the line number in the new version of the file, counted from the `+N` of its `@@ -M,K +N,L @@` hunk header over context (` `) and added (`+`) lines only. Removed (`-`) lines have no new-file number and can't carry a comment.
- `start_line` is optional and marks a range ending at `line`. Both lines must be in the same hunk.
- A finding whose lines aren't commentable is not lost: it is appended to the review body with its location.

The review is always posted as a comment. It never approves and never requests changes; a maintainer makes that call.
