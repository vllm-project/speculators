"""Sliding window attention is a no-op for performance on dense-mask backends."""

import pytest

from speculators.train.config import TrainConfig

_EXPECTED = "has no performance benefit"


def _resolve(extra: list[str]) -> None:
    TrainConfig.resolve(["--verifier-name-or-path", "dummy", *extra])


def _matches(recwarn) -> list:
    return [w for w in recwarn if _EXPECTED in str(w.message)]


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_warns_on_dense_mask_backends(impl, recwarn):
    _resolve(["--speculator-type", "dspark", "--draft-attn-impl", impl])

    matches = _matches(recwarn)
    assert len(matches) == 1, f"expected exactly one warning, got {len(matches)}"
    assert impl in str(matches[0].message)


def test_no_warning_with_flex(recwarn):
    _resolve(
        ["--speculator-type", "dspark", "--draft-attn-impl", "simple_flex_attention"]
    )

    assert not _matches(recwarn)


def test_no_warning_when_every_layer_is_full_attention(recwarn):
    """Nothing uses the window, so there is no lost benefit to report."""
    _resolve(
        [
            "--speculator-type", "dspark",
            "--draft-attn-impl", "sdpa",
            "--num-layers", "5",
            "--full-attention-indices", "0", "1", "2", "3", "4",
        ]
    )  # fmt: skip

    assert not _matches(recwarn)


def test_warns_when_only_some_layers_are_full_attention(recwarn):
    _resolve(
        [
            "--speculator-type", "dspark",
            "--draft-attn-impl", "sdpa",
            "--num-layers", "5",
            "--full-attention-indices", "4",
        ]
    )  # fmt: skip

    assert _matches(recwarn)
