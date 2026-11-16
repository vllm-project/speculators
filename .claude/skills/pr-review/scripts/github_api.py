"""Minimal read-only GitHub REST client shared by the pr-review scripts.

Uses curl with its config on stdin, so a token never reaches argv. The token
comes from GITHUB_TOKEN (or GITHUB_REVIEW_TOKEN, the name the review bot
uses); public repositories also work unauthenticated, at a lower rate limit.
"""

from __future__ import annotations

import json
import os
import subprocess

API = "https://api.github.com"
PER_PAGE = 100


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_REVIEW_TOKEN")


def get_raw(path: str, accept: str = "application/vnd.github+json") -> bytes:
    config = [
        f'url = "{API}{path}"',
        f'header = "Accept: {accept}"',
        "silent",
        "show-error",
        "fail",
        "retry = 3",
        "max-time = 30",
    ]
    if token := _token():
        config.append(f'header = "Authorization: Bearer {token}"')
    proc = subprocess.run(
        ["curl", "--config", "-"],  # noqa: S607
        input="\n".join(config).encode(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip() or proc.returncode
        raise SystemExit(f"GET {path} failed: {detail}")
    return proc.stdout


def get(path: str):
    try:
        return json.loads(get_raw(path))
    except ValueError as exc:
        raise SystemExit(f"GET {path} returned no JSON: {exc}") from exc


def paginate(path: str) -> list:
    out: list = []
    sep = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = get(f"{path}{sep}per_page={PER_PAGE}&page={page}")
        if not isinstance(batch, list) or not batch:
            return out
        out.extend(batch)
        if len(batch) < PER_PAGE:
            return out
        page += 1
