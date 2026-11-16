"""Unit tests for the pr-review skill's scripts (.claude/skills/pr-review/scripts)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

pr_state = importlib.import_module("pr_state")
review_payload = importlib.import_module("review_payload")
gather_context = importlib.import_module("gather_context")

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = (REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
UNTICKED = "- [ ] I (a human) have written or reviewed the code in this pr"
HEAD = "a" * 40


def _ticked(mark: str = "x") -> str:
    return TEMPLATE.replace(UNTICKED, UNTICKED.replace("[ ]", f"[{mark}]"))


def _pr(**overrides) -> dict:
    pr = {
        "number": 7,
        "state": "open",
        "draft": False,
        "body": _ticked(),
        "head": {"sha": HEAD},
        "base": {"repo": {"full_name": "vllm-project/speculators"}},
    }
    pr.update(overrides)
    return pr


# pr_state ---------------------------------------------------------------------


def test_checkbox_regex_matches_the_mergify_close_rule():
    rules = yaml.safe_load((REPO_ROOT / ".github" / "mergify.yml").read_text())
    conditions = next(
        rule["conditions"]
        for rule in rules["pull_request_rules"]
        if rule["name"].startswith("close PRs without the human-review checkbox")
    )
    body_condition = next(c for c in conditions if str(c).startswith("-body ~= "))
    assert body_condition.removeprefix("-body ~= ") == pr_state.HUMAN_REVIEW_RE


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (TEMPLATE, False),
        (_ticked("x"), True),
        (_ticked("X").replace("\n", "\r\n"), True),
        ("## Purpose\nNo checklist at all.\n", False),
        (None, False),
        (TEMPLATE.replace("- [ ] The purpose", "- [x] The purpose"), False),
        (f"<!-- {UNTICKED.replace('[ ]', '[x]')} -->\n{TEMPLATE}", False),
    ],
    ids=[
        "unticked",
        "ticked",
        "ticked-X-crlf",
        "no-checklist",
        "empty",
        "other-box",
        "inside-html-comment",
    ],
)
def test_human_reviewed(body, expected):
    assert pr_state.human_reviewed(body) is expected


@pytest.mark.parametrize(
    ("pr", "reviews", "reason"),
    [
        (_pr(), [], None),
        (_pr(state="closed"), [], "closed or merged"),
        (_pr(draft=True), [], "draft"),
        (_pr(body=TEMPLATE), [], "box is unticked"),
        (
            _pr(),
            [{"user": {"login": "bot"}, "state": "COMMENTED", "commit_id": HEAD}],
            "already reviewed",
        ),
        (
            _pr(),
            [{"user": {"login": "bot"}, "state": "PENDING", "commit_id": HEAD}],
            None,
        ),
        (
            _pr(),
            [{"user": {"login": "bot"}, "state": "COMMENTED", "commit_id": "b" * 40}],
            None,
        ),
        (
            _pr(),
            [{"user": {"login": "someone"}, "state": "COMMENTED", "commit_id": HEAD}],
            None,
        ),
    ],
    ids=[
        "eligible",
        "closed",
        "draft",
        "unticked",
        "reviewed-at-head",
        "pending-review",
        "reviewed-old-head",
        "other-reviewer",
    ],
)
def test_assess(pr, reviews, reason):
    state = pr_state.assess(pr, reviews, "bot")
    assert state["eligible"] is (reason is None)
    if reason is None:
        assert state["reason"] is None
    else:
        assert reason in state["reason"]


def test_assess_requires_head_sha():
    with pytest.raises(SystemExit):
        pr_state.assess(_pr(head={}), [], "bot")


# review_payload ---------------------------------------------------------------

DIFF = """\
diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1,3 +1,4 @@
 one
-two
+TWO
+three
 four
@@ -20,2 +21,2 @@
 twenty
-old
+new
\\ No newline at end of file
diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-x
"""


def test_commentable_lines():
    lines = review_payload.commentable_lines(DIFF)
    assert set(lines) == {"pkg/a.py"}
    assert lines["pkg/a.py"] == {1: 1, 2: 1, 3: 1, 4: 1, 21: 2, 22: 2}


def _review(*findings, summary="Looks reasonable.") -> dict:
    return {"version": 1, "pr": 7, "summary": summary, "findings": list(findings)}


def _build(review: dict, byline: str = ""):
    return review_payload.build_payload(review, _pr(), DIFF, HEAD, byline)


def test_payload_is_always_a_comment_on_the_reviewed_head():
    payload, _ = _build(
        _review({"confidence": 90, "body": "b", "file": "pkg/a.py", "line": 2})
    )
    assert payload["event"] == "COMMENT"
    assert payload["commit_id"] == HEAD


def test_low_confidence_findings_are_dropped_and_nothing_left_means_no_review():
    payload, counts = _build(_review({"confidence": 79, "body": "weak"}))
    assert payload is None
    assert counts["dropped"] == 1


def test_anchoring():
    payload, counts = _build(
        _review(
            {"confidence": 95, "body": "single", "file": "pkg/a.py", "line": 3},
            {
                "confidence": 90,
                "body": "range",
                "file": "pkg/a.py",
                "start_line": 2,
                "line": 4,
            },
            {
                "confidence": 90,
                "body": "cross-hunk",
                "file": "pkg/a.py",
                "start_line": 4,
                "line": 21,
            },
            {"confidence": 90, "body": "outside-diff", "file": "pkg/a.py", "line": 10},
            {"confidence": 85, "body": "pr-wide"},
            {"confidence": 50, "body": "dropped"},
        )
    )
    assert [c["body"] for c in payload["comments"]] == ["single", "range"]
    single, ranged = payload["comments"]
    assert single == {"path": "pkg/a.py", "line": 3, "side": "RIGHT", "body": "single"}
    assert ranged["start_line"] == 2
    assert ranged["start_side"] == "RIGHT"
    assert "- cross-hunk (`pkg/a.py:21`)" in payload["body"]
    assert "- outside-diff (`pkg/a.py:10`)" in payload["body"]
    assert "- pr-wide" in payload["body"].splitlines()
    assert "dropped" not in payload["body"]
    assert counts == {"dropped": 1, "inline": 2, "in_body": 3}


def test_footer_links_the_skill_at_its_commit():
    payload, _ = _build(_review({"confidence": 90, "body": "b"}), byline="by me")
    footer = payload["body"].splitlines()[-1]
    assert footer.startswith("<sub>by me · Review criteria:")
    assert (
        f"https://github.com/vllm-project/speculators/tree/{HEAD}/.claude/skills/pr-review"
        in footer
    )


@pytest.mark.parametrize(
    "review",
    [
        {"pr": 7, "summary": "s", "findings": []},
        {"version": 1, "pr": 7, "findings": []},
        {
            "version": 1,
            "pr": 7,
            "summary": "s",
            "findings": [{"confidence": "high", "body": "b"}],
        },
        {
            "version": 1,
            "pr": 7,
            "summary": "s",
            "findings": [{"confidence": 90, "body": " "}],
        },
    ],
    ids=["no-version", "no-summary", "non-int-confidence", "empty-body"],
)
def test_malformed_review_is_refused(review):
    with pytest.raises(SystemExit):
        _build(review)


def test_review_schema_accepts_the_documented_example():
    schema = json.loads(
        (
            REPO_ROOT / ".claude/skills/pr-review/references/review.schema.json"
        ).read_text()
    )
    assert set(schema["required"]) == {"version", "pr", "summary", "findings"}
    assert (
        schema["properties"]["version"]["const"] == review_payload.REVIEW_FORMAT_VERSION
    )


# gather_context ---------------------------------------------------------------


def test_gather_writes_every_context_file(tmp_path, monkeypatch):
    monkeypatch.setattr(gather_context, "get", lambda path: {"path": path})
    monkeypatch.setattr(gather_context, "get_raw", lambda path, accept: b"diff")
    monkeypatch.setattr(gather_context, "paginate", lambda path: [path])
    gather_context.gather("o/r", "7", tmp_path)
    assert json.loads((tmp_path / "pr.json").read_text()) == {
        "path": "/repos/o/r/pulls/7"
    }
    assert (tmp_path / "pr.diff").read_bytes() == b"diff"
    assert json.loads((tmp_path / "issue_comments.json").read_text()) == [
        "/repos/o/r/issues/7/comments"
    ]
    for name in ("files.json", "reviews.json", "review_comments.json"):
        assert (tmp_path / name).exists()
