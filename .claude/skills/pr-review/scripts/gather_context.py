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
  stack.json            open PRs by the same author whose head is in this PR's
                        history, nearest first, and whether own.diff was
                        written ({"parents": [], "own_diff": false} when none)
and, for a stacked PR only:
  own.diff              this PR's changes on top of its nearest parent, unless
                        the branch has merged the default branch in since
  stack/<n>/            each parent's pr.diff, reviews.json,
                        review_comments.json and issue_comments.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from github_api import get, get_raw, paginate

DIFF = "application/vnd.github.diff"


def find_parents(repo: str, pr: dict) -> list[dict]:
    """Open PRs by the PR's author whose head commit is an ancestor of its head.

    Each has its own branch on main, so this PR's diff includes theirs. Nearest
    (fewest commits between it and this head) first.
    """
    head, author = pr["head"]["sha"], pr["user"]["login"]
    parents = []
    for other in paginate(f"/repos/{repo}/pulls?state=open"):
        if other["number"] == pr["number"] or other["user"]["login"] != author:
            continue
        cmp = get(f"/repos/{repo}/compare/{other['head']['sha']}...{head}?per_page=1")
        if cmp.get("status") == "ahead":
            parents.append(
                {
                    "number": other["number"],
                    "title": other["title"],
                    "head_sha": other["head"]["sha"],
                    "commits_on_top": cmp["ahead_by"],
                }
            )
    return sorted(parents, key=lambda p: p["commits_on_top"])


def _merge_base(repo: str, branch: str, sha: str) -> str:
    cmp = get(f"/repos/{repo}/compare/{branch}...{sha}?per_page=1")
    return cmp["merge_base_commit"]["sha"]


def gather(repo: str, number: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_path = f"/repos/{repo}/pulls/{number}"
    pr = get(pr_path)
    diff = get_raw(pr_path, accept=DIFF)
    # review_payload anchors comments on the diff and posts them against
    # pr.json's head, so both must describe the same commit.
    if get(pr_path)["head"]["sha"] != pr["head"]["sha"]:
        raise SystemExit(
            f"gather_context: #{number} was pushed to during the fetch; rerun"
        )
    (out_dir / "pr.json").write_text(json.dumps(pr, indent=1))
    (out_dir / "pr.diff").write_bytes(diff)
    (out_dir / "files.json").write_text(
        json.dumps(paginate(f"{pr_path}/files"), indent=1)
    )
    _discussion(repo, number, out_dir)

    parents = find_parents(repo, pr)
    head = pr["head"]["sha"]
    own_diff = False
    if parents:
        nearest = parents[0]
        # The comparison from the parent also holds whatever was merged into
        # this branch from main since. Such a merge moves the branch's merge
        # base with main past the parent's. (The PR's commit list can't tell:
        # when the PR is based on its parent's branch, it lists the merged-in
        # commits too.)
        main = pr["base"]["repo"]["default_branch"]
        if _merge_base(repo, main, head) == _merge_base(
            repo, main, nearest["head_sha"]
        ):
            (out_dir / "own.diff").write_bytes(
                get_raw(
                    f"/repos/{repo}/compare/{nearest['head_sha']}...{head}", accept=DIFF
                )
            )
            own_diff = True
        for parent in parents:
            parent_dir = out_dir / "stack" / str(parent["number"])
            _discussion(repo, str(parent["number"]), parent_dir)
            # What the parent changes: what to leave to its review when there
            # is no own.diff to go by.
            (parent_dir / "pr.diff").write_bytes(
                get_raw(f"/repos/{repo}/pulls/{parent['number']}", accept=DIFF)
            )
    (out_dir / "stack.json").write_text(
        json.dumps({"parents": parents, "own_diff": own_diff}, indent=1)
    )


def _discussion(repo: str, number: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_path = f"/repos/{repo}/pulls/{number}"
    lists = {
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
