import pytest

from examples.quantization.fp8 import IGNORE, main


@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "FP8_BLOCK"])
def test_fp8_main_passes_quantization_arguments(tmp_path, scheme):
    calls = []

    def fake_ptq(**kwargs):
        calls.append(kwargs)

    output_dir = tmp_path / "output"
    main(
        [
            "--model",
            "model",
            "--output-dir",
            str(output_dir),
            "--scheme",
            scheme,
        ],
        ptq=fake_ptq,
    )

    assert calls == [
        {
            "model_stub": "model",
            "save_directory": str(output_dir),
            "scheme": scheme,
            "ignore": IGNORE,
        }
    ]


def test_fp8_main_refuses_existing_output(tmp_path):
    called = False

    def fake_ptq(**kwargs):
        nonlocal called
        called = True

    with pytest.raises(FileExistsError, match="Output already exists"):
        main(
            ["--model", "model", "--output-dir", str(tmp_path)],
            ptq=fake_ptq,
        )

    assert not called
