#!/usr/bin/env python3
"""Fetch everything the review rubric reads about one pull request.

usage: gather_context.py <owner/repo> <number> <out_dir>

Writes into out_dir:
  pr.json               the pull request (title, body, head sha, labels, ...)
  pr.diff               the full diff
  files.json            changed files with per-file stats
  reviews.json          submitted reviews
  review_comments.json  inline review comments
  issue_comments.json   the conversation thread
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from github_api import get, get_raw, paginate


def gather(repo: str, number: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_path = f"/repos/{repo}/pulls/{number}"
    pr = get(pr_path)
    diff = get_raw(pr_path, accept="application/vnd.github.diff")
    # review_payload anchors comments on the diff and posts them against
    # pr.json's head, so both must describe the same commit.
    if get(pr_path)["head"]["sha"] != pr["head"]["sha"]:
        raise SystemExit(
            f"gather_context: #{number} was pushed to during the fetch; rerun"
        )
    (out_dir / "pr.json").write_text(json.dumps(pr, indent=1))
    (out_dir / "pr.diff").write_bytes(diff)
    lists = {
        "files.json": f"{pr_path}/files",
        "reviews.json": f"{pr_path}/reviews",
        "review_comments.json": f"{pr_path}/comments",
        "issue_comments.json": f"/repos/{repo}/issues/{number}/comments",
    }
    for name, path in lists.items():
        (out_dir / name).write_text(json.dumps(paginate(path), indent=1))


def main() -> None:
    if len(sys.argv) != 4:  # noqa: PLR2004
        raise SystemExit("usage: gather_context.py <owner/repo> <number> <out_dir>")
    repo, number, out_dir = sys.argv[1:]
    if not number.isdigit():
        raise SystemExit(f"gather_context: {number!r} is not a pull request number")
    gather(repo, number, Path(out_dir))


if __name__ == "__main__":
    main()
