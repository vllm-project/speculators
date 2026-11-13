import sys
from types import SimpleNamespace

import pytest

from examples.quantization.fp8 import IGNORE, parse_args, quantize


@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "FP8_BLOCK"])
def test_fp8_quantize_passes_quantization_arguments(tmp_path, monkeypatch, scheme):
    calls = []

    def fake_ptq(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules, "llmcompressor", SimpleNamespace(model_free_ptq=fake_ptq)
    )
    output_dir = tmp_path / "output"
    args = parse_args(
        [
            "--model",
            "model",
            "--output-dir",
            str(output_dir),
            "--scheme",
            scheme,
        ]
    )

    quantize(args)

    assert calls == [
        {
            "model_stub": "model",
            "save_directory": str(output_dir),
            "scheme": scheme,
            "ignore": IGNORE,
        }
    ]


def test_fp8_quantize_refuses_existing_output(tmp_path, monkeypatch):
    called = False

    def fake_ptq(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setitem(
        sys.modules, "llmcompressor", SimpleNamespace(model_free_ptq=fake_ptq)
    )
    args = parse_args(["--model", "model", "--output-dir", str(tmp_path)])

    with pytest.raises(FileExistsError, match="Output already exists"):
        quantize(args)

    assert not called
