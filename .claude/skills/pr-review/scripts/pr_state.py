#!/usr/bin/env python3
"""Decide whether a pull request should be reviewed.

usage: pr_state.py <owner/repo> <number> <reviewer>

Prints one JSON object: number, head_sha, open, draft, human_reviewed,
reviewed_at_head, eligible, and reason (why it is not eligible, or null).
Exits non-zero when GitHub cannot be read: callers treat an unreadable state
as "do not review", never as a pass.
"""

from __future__ import annotations

import json
import re
import sys

from github_api import get, paginate

# Must stay identical to the `-body ~=` condition of the human-review close
# rule in .github/mergify.yml; a unit test enforces this.
HUMAN_REVIEW_RE = r"(?i)[-*]\s*\[x\]\s*I \(a human\) have written or reviewed"
# Mergify matches `body` with HTML comments stripped; do the same.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def human_reviewed(body: str | None) -> bool:
    """True when the "I (a human) have written or reviewed" box is ticked."""
    return bool(re.search(HUMAN_REVIEW_RE, _HTML_COMMENT.sub("", body or "")))


def assess(pr: dict, reviews: list[dict], reviewer: str) -> dict:
    head_sha = (pr.get("head") or {}).get("sha") or ""
    if not head_sha:
        raise SystemExit(f"pr_state: #{pr.get('number')} has no head sha")
    state = {
        "number": pr.get("number"),
        "head_sha": head_sha,
        "open": pr.get("state") == "open",
        "draft": bool(pr.get("draft")),
        "human_reviewed": human_reviewed(pr.get("body")),
        "reviewed_at_head": any(
            (r.get("user") or {}).get("login") == reviewer
            and r.get("state") != "PENDING"
            and r.get("commit_id") == head_sha
            for r in reviews
        ),
    }
    if not state["open"]:
        reason = "closed or merged"
    elif state["draft"]:
        reason = "draft"
    elif not state["human_reviewed"]:
        reason = 'the "I (a human) have written or reviewed the code" box is unticked'
    elif state["reviewed_at_head"]:
        reason = f"{reviewer} already reviewed the current head"
    else:
        reason = None
    state["eligible"] = reason is None
    state["reason"] = reason
    return state


def main() -> None:
    if len(sys.argv) != 4:  # noqa: PLR2004
        raise SystemExit("usage: pr_state.py <owner/repo> <number> <reviewer>")
    repo, number, reviewer = sys.argv[1:]
    if not number.isdigit():
        raise SystemExit(f"pr_state: {number!r} is not a pull request number")
    pr = get(f"/repos/{repo}/pulls/{number}")
    reviews = paginate(f"/repos/{repo}/pulls/{number}/reviews")
    sys.stdout.write(json.dumps(assess(pr, reviews, reviewer)) + "\n")


if __name__ == "__main__":
    main()
