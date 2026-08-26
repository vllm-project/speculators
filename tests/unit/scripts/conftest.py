"""Put scripts/ on sys.path so tests can import launch_vllm and _provenance directly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
