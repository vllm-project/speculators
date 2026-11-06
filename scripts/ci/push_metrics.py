#!/usr/bin/env python3
"""Push CI metrics to the ``metrics-data`` orphan branch.

Stores JSON arrays on an orphan branch readable by Grafana's
Infinity data-source plugin via raw GitHub URLs.

Examples::

    # One-time setup — create the orphan branch
    python scripts/ci/push_metrics.py init

    # Push a single JSON record
    python scripts/ci/push_metrics.py push record.json --target acceptance.json

    # Push all .json files collected during a CI run
    python scripts/ci/push_metrics.py push-dir /tmp/ci-metrics --target acceptance.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BRANCH = "metrics-data"
MAX_RETRIES = 5
BOT_NAME = "speculators-bot"
BOT_EMAIL = "speculators-bot@users.noreply.github.com"


def _git(
    args: list[str],
    cwd: str | Path,
    *,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _remote_url(repo_dir: Path) -> str:
    url = _git(["remote", "get-url", "origin"], repo_dir).stdout.strip()
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://"):
        url = url.replace("https://", f"https://x-access-token:{token}@", 1)
    return url


def _configure_bot(cwd: str | Path) -> None:
    _git(["config", "user.name", BOT_NAME], cwd)
    _git(["config", "user.email", BOT_EMAIL], cwd)


# ---- subcommands -----------------------------------------------------------


def cmd_init(_args: argparse.Namespace) -> None:
    """Create the orphan metrics-data branch with an initial README."""
    remote_url = _remote_url(_repo_root())

    with tempfile.TemporaryDirectory() as tmp:
        _git(["init", "."], tmp)
        _configure_bot(tmp)
        _git(["checkout", "--orphan", BRANCH], tmp)

        Path(tmp, "README.md").write_text(
            "# Speculators CI Metrics\n\n"
            "Orphan branch storing CI benchmark metrics as JSONL.\n"
            "Consumed by Grafana via the Infinity data-source plugin.\n",
        )

        _git(["add", "README.md"], tmp)
        _git(["commit", "-m", "Initialize metrics-data branch"], tmp)
        _git(["remote", "add", "origin", remote_url], tmp)
        _git(["push", "origin", BRANCH], tmp, timeout=60)

    print(f"Created orphan branch '{BRANCH}'")


def _push_records(records: list[dict], target: str, branch: str) -> None:
    """Append *records* to the JSON array in *target* on *branch*."""
    remote_url = _remote_url(_repo_root())

    for attempt in range(MAX_RETRIES):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _git(
                    [
                        "clone",
                        "--branch",
                        branch,
                        "--single-branch",
                        "--depth",
                        "1",
                        remote_url,
                        ".",
                    ],
                    tmp,
                    timeout=60,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Branch '{branch}' not found. Run 'push_metrics.py init' first."
                ) from exc

            _configure_bot(tmp)

            target_path = Path(tmp) / target
            target_path.parent.mkdir(parents=True, exist_ok=True)

            existing: list[dict] = []
            if target_path.exists():
                existing = json.loads(target_path.read_text())

            existing.extend(records)
            target_path.write_text(json.dumps(existing))

            _git(["add", target], tmp)

            if not _git(["status", "--porcelain"], tmp).stdout.strip():
                print("No changes to push.")
                return

            n = len(records)
            _git(
                ["commit", "-m", f"Add {n} record(s) to {target}"],
                tmp,
            )

            result = _git(
                ["push", "origin", branch],
                tmp,
                check=False,
                timeout=60,
            )
            if result.returncode == 0:
                print(f"Pushed {n} record(s) to {branch}:{target}")
                return

            if attempt < MAX_RETRIES - 1:
                print(
                    f"Push conflict (attempt {attempt + 1}/{MAX_RETRIES}), retrying…",
                )
            else:
                raise RuntimeError(
                    f"Failed to push after {MAX_RETRIES} attempts:\n{result.stderr}",
                )


def _load_records(path: Path) -> list[dict]:
    """Load a JSON file as a list of records (object or array)."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return [data]


def cmd_push(args: argparse.Namespace) -> None:
    """Push a single JSON file (object or array of objects)."""
    if args.record_file == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.record_file).read_text()
    data = json.loads(raw)
    records = data if isinstance(data, list) else [data]
    _push_records(records, args.target, args.branch)


def cmd_push_dir(args: argparse.Namespace) -> None:
    """Push every ``*.json`` file in a directory as one commit."""
    d = Path(args.directory)
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    for p in sorted(d.glob("*.json")):
        records.extend(_load_records(p))

    if not records:
        print("No .json files found.")
        return

    _push_records(records, args.target, args.branch)


# ---- CLI -------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Push CI metrics to the metrics-data orphan branch",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help=f"Create the '{BRANCH}' orphan branch")

    sp = sub.add_parser("push", help="Push a single JSON record")
    sp.add_argument("record_file", help="Path to a JSON file, or '-' for stdin")
    sp.add_argument(
        "--target",
        required=True,
        help="JSON file on the metrics branch (e.g. acceptance.json)",
    )
    sp.add_argument("--branch", default=BRANCH)

    sp = sub.add_parser("push-dir", help="Push all .json files from a directory")
    sp.add_argument("directory", help="Directory containing .json record files")
    sp.add_argument(
        "--target",
        required=True,
        help="JSON file on the metrics branch (e.g. acceptance.json)",
    )
    sp.add_argument("--branch", default=BRANCH)

    args = p.parse_args()
    {"init": cmd_init, "push": cmd_push, "push-dir": cmd_push_dir}[args.command](args)


if __name__ == "__main__":
    main()
