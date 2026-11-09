"""Unit tests for scripts/loss_head_benchmark.py (device parts skip without a GPU)."""

from __future__ import annotations

import json

import pytest
from loss_head_benchmark import main  # type: ignore[import-not-found]

from tests.conftest import requires_cuda


@pytest.mark.parametrize(
    "bad",
    [
        "--implementation=fsued",
        "--model=dflash2",
        "--weighting=dpase",
        "--loss=kl",
        "--rows=4",
    ],
)
def test_run_rejects_unknown_options_before_measuring(bad):
    # Tiny shapes: if validation regresses this measures fast, not GB. The
    # bad option goes last so it overrides the defaults given here.
    tiny = ["--rows=16", "--vocab=32", "--hidden=8", "--iters=1", "--warmup=0"]
    with pytest.raises(SystemExit, match="unknown|invalid|smaller"):
        main(["run", *tiny, bad])


@requires_cuda
def test_run_smoke(tmp_path):
    out = tmp_path / "run.json"
    payload = main(
        [
            "run",
            "--rows=256",
            "--vocab=1000",
            "--hidden=64",
            "--iters=1",
            "--warmup=1",
            "--weighting=fixed-exp-decay,dpace",
            f"--output={out}",
        ]
    )
    assert json.loads(out.read_text()) == payload
    assert len(payload["results"]) == 4  # 2 models x 2 weightings
    for result in payload["results"]:
        assert result["status"] == "ok"
        assert result["peak_above_inputs_mb"] > 0
