"""Sync src/speculators/provenance.py → scripts/_provenance.py.

Run with no args to regenerate; run with --check to verify the copy is current.
Wired into `make style` (regenerate) and `make quality` (check).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "speculators" / "provenance.py"
DST = REPO_ROOT / "scripts" / "_provenance.py"

BANNER = """\
# ============================================================================
# AUTO-GENERATED — do not edit directly.
# Source of truth: src/speculators/provenance.py
# Regenerate with: make style   (or: python scripts/sync_provenance.py)
# ============================================================================
"""


def _generated(src_text: str) -> str:
    return BANNER + src_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if scripts/_provenance.py is out of sync",
    )
    args = parser.parse_args()

    src_text = SRC.read_text()
    expected = _generated(src_text)

    if args.check:
        if not DST.exists():
            print(
                f"ERROR: {DST} does not exist. Run `make style` to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)
        actual = DST.read_text()
        if actual != expected:
            print(
                f"ERROR: {DST} is out of sync with {SRC}.\n"
                "Run `make style` (or: python scripts/sync_provenance.py) to update.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK: {DST} is in sync.")
    else:
        fd, tmp = tempfile.mkstemp(
            dir=DST.parent, prefix="._provenance_", suffix=".tmp"
        )
        tmp_path = Path(tmp)
        try:
            with open(fd, "w") as f:
                f.write(expected)
            tmp_path.replace(DST)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"Synced {SRC} → {DST}")


if __name__ == "__main__":
    main()
