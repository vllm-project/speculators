#!/usr/bin/env python3
"""Turn a drafted review.json into a GitHub "create review" payload.

usage: review_payload.py <review.json> <context_dir> --skill-ref <sha>
                         [--byline <text>]

context_dir is the output of gather_context.py (pr.json and pr.diff are
read). Prints the payload JSON on stdout, ready for
`gh api repos/<owner>/<repo>/pulls/<n>/reviews -X POST --input -`.

Enforced here so every caller gets the same rules:
- the draft must be for the PR in the context directory;
- findings below MIN_CONFIDENCE are dropped;
- the event is always COMMENT (never APPROVE or REQUEST_CHANGES);
- a finding becomes an inline comment only if its lines are commentable
  right-side lines of one diff hunk (GitHub rejects the whole review
  otherwise); any other finding goes into the review body;
- a finding's `detail` is posted under its body in a collapsed section;
- `path:line` references to changed files become permalinks at the
  reviewed head;
- the body ends with a link to the skill at the commit that produced it.

Exits with status 3 and prints nothing when no finding survives: clean PRs
get no review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from urllib.parse import quote

REVIEW_FORMAT_VERSION = 1
MIN_CONFIDENCE = 80
EXIT_NOTHING_TO_POST = 3
SKILL_PATH = ".claude/skills/pr-review"
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


_ESCAPES = {"n": "\n", "t": "\t"}


def _unquote(name: str) -> str:
    """A path as git writes it in a diff header. Git quotes names with unusual
    characters C-style, with non-ASCII bytes as octal escapes."""
    target = name.strip()
    if len(target) > 1 and target[0] == target[-1] == '"':
        raw = bytearray()
        for part in re.split(r"(\\(?:[0-7]{3}|.))", target[1:-1]):
            if re.fullmatch(r"\\[0-7]{3}", part):
                raw.append(int(part[1:], 8))
            elif part.startswith("\\"):
                raw += _ESCAPES.get(part[1], part[1]).encode()
            else:
                raw += part.encode()
        target = raw.decode("utf-8", errors="replace")
    return target


def _target_path(header: str) -> str | None:
    """The new-file path of a `+++ ` header, or None for /dev/null."""
    target = _unquote(header)
    return target[2:] if target.startswith("b/") else None


def commentable_lines(diff: str) -> dict[str, dict[int, int]]:
    """Map path -> {new-file line number: hunk index} for added/context lines."""
    out: dict[str, dict[int, int]] = {}
    path: str | None = None
    line = hunk = 0
    in_header = False
    # Only newlines end a diff line; str.splitlines() also splits on \f, \u2028
    # and the like, which can appear inside a line's text.
    for raw in diff.split("\n"):
        if raw.startswith("diff --git "):
            path, line, in_header = None, 0, True
        elif in_header and raw.startswith("rename to "):
            # A pure rename has no `+++` header but is still a changed file;
            # it has no commentable lines unless a `+++` header follows.
            out.setdefault(_unquote(raw[len("rename to ") :]), {})
        elif in_header and raw.startswith("+++ "):
            # Only a file's header carries `+++ `: an added line whose text
            # starts with "++ " looks the same once it's in a hunk.
            path = _target_path(raw[4:])
            if path is not None:
                out.setdefault(path, {})
        elif raw.startswith("@@"):
            in_header = False
            match = _HUNK.match(raw)
            line = int(match.group(1)) if match else 0
            hunk += 1
        elif path is None or line == 0 or raw.startswith("\\"):
            continue
        elif raw.startswith(("+", " ")):
            out[path][line] = hunk
            line += 1
    return out


def _validate(review: dict) -> None:
    if review.get("version") != REVIEW_FORMAT_VERSION:
        raise SystemExit(
            f"review_payload: review.json version {review.get('version')!r} "
            f"is not {REVIEW_FORMAT_VERSION}"
        )
    if not isinstance(review.get("summary"), str):
        raise SystemExit("review_payload: review.json needs a string `summary`")
    for i, finding in enumerate(review.get("findings", [])):
        if not isinstance(finding.get("confidence"), int):
            raise SystemExit(f"review_payload: finding {i} needs an int `confidence`")
        if not (finding.get("body") or "").strip():
            raise SystemExit(f"review_payload: finding {i} needs a `body`")
        if not isinstance(finding.get("detail", ""), str):
            raise SystemExit(f"review_payload: finding {i} has a non-string `detail`")


def _detail_block(finding: dict) -> str:
    """The finding's detail as a collapsed section, or "" when it has none.

    The blank lines make GitHub render the inside as Markdown.
    """
    detail = (finding.get("detail") or "").strip()
    if not detail:
        return ""
    return f"\n\n<details><summary>Details</summary>\n\n{detail}\n\n</details>"


def _inline_comment(finding: dict, anchors: dict[str, dict[int, int]]) -> dict | None:
    path, line = finding.get("file"), finding.get("line")
    if not isinstance(path, str) or not isinstance(line, int):
        return None
    lines = anchors.get(path, {})
    if line not in lines:
        return None
    body = finding["body"].strip() + _detail_block(finding)
    comment = {"path": path, "line": line, "side": "RIGHT", "body": body}
    start = finding.get("start_line")
    if isinstance(start, int) and start != line:
        if start > line or lines.get(start) != lines[line]:
            return None
        comment.update(start_line=start, start_side="RIGHT")
    return comment


# `path:12` or `path:12-15`; `path` may be a suffix such as a bare file name.
_REF = r"((?:[\w.-]+/)*[\w.-]+\.\w+):(\d+)(?:-(\d+))?"
_BARE_REF = re.compile(r"(?<![\w\[(/`.-])" + _REF + r"(?![\w\]`-])")
_CODE_REF = re.compile("`" + _REF + "`")
_FENCE = re.compile(r"(```.*?```|~~~.*?~~~)", re.S)
# An existing Markdown link, whose text may itself be a code span.
_LINK = re.compile(r"(\[[^\]\n]*\]\([^)\n]*\))")
# A double-backtick span first, so its inner backticks don't start a single one.
_CODE_SPAN = re.compile(r"(``.+?``|`[^`\n]+`)")


def link_refs(text: str, paths: list[str], repo: str, sha: str) -> str:
    """Turn `path:line` references to changed files into permalinks at `sha`.

    A reference resolves when it names exactly one changed file, by its full
    path or a path suffix; anything else is left as written. Fenced blocks,
    existing links and code spans other than a bare reference are not
    touched.
    """

    def resolve(match: re.Match) -> str:
        ref, start, end = match.groups()
        hits = [p for p in paths if p == ref or p.endswith("/" + ref)]
        if len(hits) != 1:
            return match.group(0)
        anchor = f"L{start}" + (f"-L{end}" if end else "")
        label = f"{ref}:{start}" + (f"-{end}" if end else "")
        url_path = quote(hits[0], safe="/")
        return f"[`{label}`](https://github.com/{repo}/blob/{sha}/{url_path}#{anchor})"

    out = []
    for i, block in enumerate(_FENCE.split(text)):
        if i % 2:  # a fenced block
            out.append(block)
            continue
        for k, chunk in enumerate(_LINK.split(block)):
            if k % 2:  # an existing link
                out.append(chunk)
                continue
            for j, piece in enumerate(_CODE_SPAN.split(chunk)):
                if j % 2:  # an inline code span
                    full = _CODE_REF.fullmatch(piece)
                    out.append(resolve(full) if full else piece)
                else:
                    out.append(_BARE_REF.sub(resolve, piece))
    return "".join(out)


def build_payload(
    review: dict, pr: dict, diff: str, skill_ref: str, byline: str = ""
) -> tuple[dict | None, dict]:
    """Return (payload or None when nothing survives, counts)."""
    _validate(review)
    if str(review.get("pr")) != str(pr.get("number")):
        raise SystemExit(
            f"review_payload: review.json is for PR {review.get('pr')!r}, "
            f"the context is PR {pr.get('number')!r}"
        )
    findings = review.get("findings", [])
    kept = [f for f in findings if f["confidence"] >= MIN_CONFIDENCE]
    counts = {"dropped": len(findings) - len(kept), "inline": 0, "in_body": 0}
    if not kept:
        return None, counts

    anchors = commentable_lines(diff)
    repo = ((pr.get("base") or {}).get("repo") or {}).get("full_name", "")
    head = (pr.get("head") or {}).get("sha", "")
    paths = list(anchors)

    def linked(text: str) -> str:
        return link_refs(text, paths, repo, head)

    comments, in_body = [], []
    for finding in kept:
        comment = _inline_comment(finding, anchors)
        if comment is not None:
            comment["body"] = linked(comment["body"])
            comments.append(comment)
            continue
        where = finding.get("file")
        if where and isinstance(finding.get("line"), int):
            where = f"{where}:{finding['line']}"
        item = f"- {finding['body'].strip()}" + (f" (`{where}`)" if where else "")
        # Indented, the collapsed section stays part of its list item.
        item += textwrap.indent(_detail_block(finding), "  ")
        in_body.append(linked(item))
    counts.update(inline=len(comments), in_body=len(in_body))

    skill_url = f"https://github.com/{repo}/tree/{skill_ref}/{SKILL_PATH}"
    footer = f"Review criteria: [`pr-review` skill @ `{skill_ref[:12]}`]({skill_url})"
    if byline:
        footer = f"{byline} · {footer}"
    # The summary is often empty: the findings then stand on their own.
    parts = [linked(review["summary"].strip()), "\n".join(in_body)]
    body = "\n\n".join([*filter(None, parts), f"<sub>{footer}</sub>"])

    payload = {
        "commit_id": head,
        "event": "COMMENT",
        "body": body,
        "comments": comments,
    }
    return payload, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("review_json", type=Path)
    parser.add_argument("context_dir", type=Path)
    parser.add_argument("--skill-ref", required=True, help="commit of the skill used")
    parser.add_argument("--byline", default="", help="who or what wrote the review")
    args = parser.parse_args()

    payload, counts = build_payload(
        json.loads(args.review_json.read_text()),
        json.loads((args.context_dir / "pr.json").read_text()),
        (args.context_dir / "pr.diff").read_text(),
        args.skill_ref,
        args.byline,
    )
    sys.stderr.write(
        f"review_payload: {counts['inline']} inline, {counts['in_body']} in body, "
        f"{counts['dropped']} below confidence {MIN_CONFIDENCE}\n"
    )
    if payload is None:
        sys.stderr.write("review_payload: no finding survives; nothing to post\n")
        raise SystemExit(EXIT_NOTHING_TO_POST)
    sys.stdout.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
