import sys
from types import SimpleNamespace

from transformers import AutoConfig

from scripts import launch_vllm


def _run_launch_vllm_dry_run(monkeypatch, capsys, args: list[str]) -> str:
    monkeypatch.setattr(sys, "argv", ["launch_vllm.py", *args])

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(lambda _: SimpleNamespace(num_hidden_layers=4)),
    )

    launch_vllm.main()
    return capsys.readouterr().out


def test_launch_vllm_disables_prefix_cache_by_default(monkeypatch, capsys):
    output = _run_launch_vllm_dry_run(monkeypatch, capsys, ["dummy", "--dry-run"])

    assert "--no-enable-prefix-caching" in output
    assert "--no-enable-chunked-prefill" in output


def test_launch_vllm_preserves_explicit_prefix_cache_arg(monkeypatch, capsys):
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run", "--", "--enable-prefix-caching"],
    )

    assert output.count("--enable-prefix-caching") == 1
    assert "--no-enable-prefix-caching" not in output
