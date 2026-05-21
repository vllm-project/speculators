import os
from collections.abc import Callable

import pytest


def requires_cadence(cadence: str | list[str]) -> Callable:
    cadence = [cadence] if isinstance(cadence, str) else cadence
    current_cadence = os.environ.get("CADENCE", "commit")

    return pytest.mark.skipif(
        (current_cadence not in cadence), reason="cadence mismatch"
    )
