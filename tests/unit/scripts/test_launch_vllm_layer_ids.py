"""launch_vllm.py rejects invalid target layer ids before building the vLLM command."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import launch_vllm  # type: ignore[import-not-found]


@pytest.mark.parametrize(
    ("num_layers", "extra_args", "ok"),
    [
        (28, [], True),  # default [2, 14, 25, 28]
        (4, [], False),  # default [2, 2, 1, 4] repeats an id
        (28, ["--target-layer-ids", "1", "9", "17"], True),
        (28, ["--target-layer-ids", "2", "14", "28"], True),  # last layer is valid
        (28, ["--target-layer-ids", "2", "2", "1"], False),
        (28, ["--target-layer-ids", "-1", "2", "3"], False),
        (28, ["--target-layer-ids", "2", "14", "29"], False),  # beyond last layer
    ],
)
def test_launch_rejects_invalid_target_layer_ids(
    num_layers, extra_args, ok, monkeypatch
):
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda *_a, **_k: SimpleNamespace(num_hidden_layers=num_layers)),
    )
    monkeypatch.setattr(
        sys, "argv", ["launch_vllm.py", "verifier", "--dry-run", *extra_args]
    )
    if ok:
        launch_vllm.main()
    else:
        with pytest.raises(ValueError, match="distinct and within"):
            launch_vllm.main()
