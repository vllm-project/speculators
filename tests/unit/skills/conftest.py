"""Put the pr-review skill's scripts/ on sys.path so tests can import them."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".claude" / "skills" / "pr-review" / "scripts"))
