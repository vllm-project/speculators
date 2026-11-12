from pathlib import Path

import pytest

from tests.e2e.utils import run_benchmark
from tests.utils import requires_cadence


@pytest.mark.regression
@requires_cadence("nightly")
def test_bench_dflash_adamw(tmp_path: Path):
    """Forward/loss-dominated: dflash + AdamW, CE loss path."""
    run_benchmark(
        tmp_path,
        speculator_type="dflash",
        extra_train_args=["--optimizer", "adamw"],
    )


@pytest.mark.regression
@requires_cadence("nightly")
def test_bench_dspark_muon(tmp_path: Path):
    """Optimizer-dominated: dspark + Muon (default), KL loss."""
    run_benchmark(
        tmp_path,
        speculator_type="dspark",
    )


@pytest.mark.regression
@requires_cadence("nightly")
def test_bench_mtp_adamw(tmp_path: Path):
    """Different architecture: mtp + AdamW, KL loss."""
    run_benchmark(
        tmp_path,
        speculator_type="mtp",
        extra_train_args=["--optimizer", "adamw"],
    )
