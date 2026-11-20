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


def test_force_lifts_only_the_already_reviewed_block():
    reviewed = [{"user": {"login": "bot"}, "state": "COMMENTED", "commit_id": HEAD}]
    assert pr_state.assess(_pr(), reviewed, "bot", force=True)["eligible"]
    for pr in (_pr(state="closed"), _pr(draft=True), _pr(body=TEMPLATE)):
        assert not pr_state.assess(pr, reviewed, "bot", force=True)["eligible"]


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


def test_an_added_line_that_looks_like_a_file_header_stays_in_its_file():
    diff = (
        "diff --git a/docs/x.md b/docs/x.md\n--- a/docs/x.md\n+++ b/docs/x.md\n"
        "@@ -1,2 +1,4 @@\n a\n+++ b/example\n+b\n c\n"
    )
    assert review_payload.commentable_lines(diff) == {
        "docs/x.md": {1: 1, 2: 1, 3: 1, 4: 1}
    }


def test_quoted_paths_and_form_feeds():
    diff = (
        'diff --git "a/d\\303\\251j\\303\\240.md" "b/d\\303\\251j\\303\\240.md"\n'
        '--- "a/d\\303\\251j\\303\\240.md"\n+++ "b/d\\303\\251j\\303\\240.md"\n'
        "@@ -1 +1,3 @@\n a\n+page\x0cbreak\n+c\n"
    )
    assert review_payload.commentable_lines(diff) == {"déjà.md": {1: 1, 2: 1, 3: 1}}


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
        {"version": 1, "pr": 8, "summary": "s", "findings": []},
        {"version": 1, "summary": "s", "findings": []},
    ],
    ids=[
        "no-version",
        "no-summary",
        "non-int-confidence",
        "empty-body",
        "other-pr",
        "no-pr",
    ],
)
def test_malformed_review_is_refused(review):
    with pytest.raises(SystemExit):
        _build(review)


def test_pr_number_may_be_a_string():
    review = _review({"confidence": 90, "body": "b"})
    review["pr"] = "7"
    payload, _ = _build(review)
    assert payload is not None


def test_the_documented_example_builds_a_review():
    doc = (REPO_ROOT / ".claude/skills/pr-review/references/output.md").read_text()
    example = json.loads(doc.split("```json\n", 1)[1].split("```", 1)[0])
    assert example["version"] == review_payload.REVIEW_FORMAT_VERSION
    pr = _pr(number=example["pr"])
    payload, counts = review_payload.build_payload(example, pr, DIFF, HEAD)
    assert payload is not None
    assert counts["dropped"] == 0
    assert counts["inline"] + counts["in_body"] == len(example["findings"])


# gather_context ---------------------------------------------------------------


class FakeGitHub:
    """Serves PR 7 by `author` (at each of `heads` in turn), the repo's other open
    PRs, and compare results: `ancestors` maps a head sha to the commits between
    it and PR 7's head, and `merge_bases` a sha to its merge base with main, the
    default branch (by default one shared by every sha)."""

    def __init__(self, *heads, others=(), ancestors=None, merge_bases=None):
        self.heads = iter(heads)
        self.others = list(others)
        self.ancestors = ancestors or {}
        self.merge_bases = merge_bases or {}

    def get(self, path):
        if path.startswith("/repos/o/r/compare/"):
            basehead = path.removeprefix("/repos/o/r/compare/").split("?")[0]
            base, head = basehead.split("...")
            if base in self.ancestors:
                return {"status": "ahead", "ahead_by": self.ancestors[base]}
            if base == "main":
                sha = self.merge_bases.get(head, "m" * 40)
                return {"status": "diverged", "merge_base_commit": {"sha": sha}}
            return {"status": "diverged", "ahead_by": 3}
        return {
            "number": 7,
            "user": {"login": "author"},
            "head": {"sha": next(self.heads)},
            "base": {"repo": {"default_branch": "main"}},
        }

    def paginate(self, path):
        if path.startswith("/repos/o/r/pulls?"):
            return self.others
        return [path]

    def install(self, monkeypatch):
        monkeypatch.setattr(gather_context, "get", self.get)
        monkeypatch.setattr(gather_context, "paginate", self.paginate)
        monkeypatch.setattr(
            gather_context, "get_raw", lambda path, accept: f"diff {path}".encode()
        )


def _other(number, login, sha):
    return {
        "number": number,
        "title": f"pr {number}",
        "user": {"login": login},
        "head": {"sha": sha},
    }


def test_gather_writes_every_context_file(tmp_path, monkeypatch):
    FakeGitHub(HEAD, HEAD).install(monkeypatch)
    gather_context.gather("o/r", "7", tmp_path)
    assert json.loads((tmp_path / "pr.json").read_text())["head"] == {"sha": HEAD}
    assert (tmp_path / "pr.diff").read_bytes() == b"diff /repos/o/r/pulls/7"
    assert json.loads((tmp_path / "issue_comments.json").read_text()) == [
        "/repos/o/r/issues/7/comments"
    ]
    for name in ("files.json", "reviews.json", "review_comments.json"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "stack.json").read_text()) == {
        "parents": [],
        "own_diff": False,
    }
    assert not (tmp_path / "own.diff").exists()
    assert not (tmp_path / "stack").exists()


def test_gather_refuses_a_push_between_metadata_and_diff(tmp_path, monkeypatch):
    FakeGitHub(HEAD, "b" * 40).install(monkeypatch)
    with pytest.raises(SystemExit, match="pushed to during the fetch"):
        gather_context.gather("o/r", "7", tmp_path)
    assert not (tmp_path / "pr.json").exists()


def test_a_stacked_pr_gets_its_own_diff_and_its_parents_discussion(
    tmp_path, monkeypatch
):
    near, far, unrelated, foreign = "1" * 40, "2" * 40, "3" * 40, "4" * 40
    FakeGitHub(
        HEAD,
        HEAD,
        others=[
            _other(5, "author", far),
            _other(6, "author", near),
            _other(8, "author", unrelated),  # diverged: not a parent
            _other(9, "someone", foreign),  # ancestor, but another author's
            _other(7, "author", HEAD),  # the PR itself
        ],
        ancestors={near: 2, far: 9, foreign: 1},
    ).install(monkeypatch)
    gather_context.gather("o/r", "7", tmp_path)
    parents = json.loads((tmp_path / "stack.json").read_text())["parents"]
    assert [(p["number"], p["commits_on_top"]) for p in parents] == [(6, 2), (5, 9)]
    assert (tmp_path / "own.diff").read_bytes() == (
        f"diff /repos/o/r/compare/{near}...{HEAD}".encode()
    )
    for n in ("5", "6"):
        assert json.loads(
            (tmp_path / "stack" / n / "issue_comments.json").read_text()
        ) == [f"/repos/o/r/issues/{n}/comments"]
        assert (tmp_path / "stack" / n / "pr.diff").read_bytes() == (
            f"diff /repos/o/r/pulls/{n}".encode()
        )


def test_no_own_diff_when_the_branch_merged_main_since_its_parent(
    tmp_path, monkeypatch
):
    parent = "1" * 40
    FakeGitHub(
        HEAD,
        HEAD,
        others=[_other(6, "author", parent)],
        ancestors={parent: 3},
        merge_bases={HEAD: "n" * 40},
    ).install(monkeypatch)
    gather_context.gather("o/r", "7", tmp_path)
    stack = json.loads((tmp_path / "stack.json").read_text())
    assert [p["number"] for p in stack["parents"]] == [6]
    assert stack["own_diff"] is False
    assert not (tmp_path / "own.diff").exists()
    assert (tmp_path / "stack" / "6" / "issue_comments.json").exists()


def test_every_listed_parent_gets_its_discussion_staged(tmp_path, monkeypatch):
    shas = [str(i) * 40 for i in range(1, 7)]
    FakeGitHub(
        HEAD,
        HEAD,
        others=[_other(10 + i, "author", s) for i, s in enumerate(shas)],
        ancestors={s: i + 1 for i, s in enumerate(shas)},
    ).install(monkeypatch)
    gather_context.gather("o/r", "7", tmp_path)
    listed = [
        p["number"]
        for p in json.loads((tmp_path / "stack.json").read_text())["parents"]
    ]
    assert len(listed) == 6
    assert sorted(int(d.name) for d in (tmp_path / "stack").iterdir()) == sorted(listed)
