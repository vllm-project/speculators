import json
import sys
from types import SimpleNamespace

import pytest
from transformers import AutoConfig

from scripts import launch_vllm


def _run_launch_vllm_dry_run(
    monkeypatch,
    capsys,
    args: list[str],
    config=None,
) -> str:
    monkeypatch.setattr(sys, "argv", ["launch_vllm.py", *args])

    config = config or SimpleNamespace(num_hidden_layers=4)
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(lambda _: config),
    )

    launch_vllm.main()
    return capsys.readouterr().out


def _extract_speculative_config(output: str) -> dict:
    command = output.split("Running command:\n", 1)[1].strip()
    payload = command.split(" --speculative_config ", 1)[1]
    raw_config = payload.split(" --kv_transfer_config ", 1)[0]
    return json.loads(raw_config)


def _extract_kv_transfer_config(output: str) -> dict:
    command = output.split("Running command:\n", 1)[1].strip()
    payload = command.split(" --kv_transfer_config ", 1)[1]
    return json.loads(payload.split(" --", 1)[0])


def test_launch_vllm_disables_prefix_cache_by_default(monkeypatch, capsys):
    """Hidden-state extraction needs full-prompt slots by default."""
    output = _run_launch_vllm_dry_run(monkeypatch, capsys, ["dummy", "--dry-run"])

    assert "--no-enable-prefix-caching" in output


def test_launch_vllm_makes_hidden_states_path_absolute(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.chdir(tmp_path)
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run", "--hidden-states-path", "runtime/hidden_states"],
    )

    kv_config = _extract_kv_transfer_config(output)
    shared_path = kv_config["kv_connector_extra_config"]["shared_storage_path"]
    assert shared_path == str(tmp_path / "runtime/hidden_states")


def test_launch_vllm_small_verifier_uses_unique_valid_default_layers(
    monkeypatch,
    capsys,
):
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run"],
        config=SimpleNamespace(num_hidden_layers=4),
    )

    hf_config = _extract_speculative_config(output)["draft_model_config"]["hf_config"]
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [1, 2, 3, 4]
    assert "--no-enable-chunked-prefill" in output


def test_launch_vllm_preserves_explicit_prefix_cache_arg(monkeypatch, capsys):
    """User-provided prefix-cache flags override extraction defaults."""
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run", "--", "--enable-prefix-caching"],
    )

    assert output.count("--enable-prefix-caching") == 1
    assert "--no-enable-prefix-caching" not in output


@pytest.mark.parametrize(
    "flag",
    ["--enable-chunked-prefill", "--no-enable-chunked-prefill"],
)
def test_launch_vllm_preserves_explicit_chunked_prefill_arg(
    monkeypatch, capsys, flag
):
    """Explicit chunked-prefill choices must not gain a contradictory default."""
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run", "--", flag],
    )

    assert output.count(flag) == 1
    opposite = (
        "--no-enable-chunked-prefill"
        if flag == "--enable-chunked-prefill"
        else "--enable-chunked-prefill"
    )
    assert opposite not in output


def test_launch_vllm_falls_back_when_text_config_is_none(monkeypatch, capsys):
    """A null nested text_config falls back to top-level text attributes."""
    config = SimpleNamespace(
        num_hidden_layers=8,
        text_config=None,
        vision_config=SimpleNamespace(),
    )

    with pytest.warns(UserWarning, match="multimodal verifier config"):
        output = _run_launch_vllm_dry_run(
            monkeypatch,
            capsys,
            ["dummy", "--dry-run"],
            config=config,
        )

    hf_config = _extract_speculative_config(output)["draft_model_config"][
        "hf_config"
    ]
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [2, 4, 5, 8]


def test_launch_vllm_flattens_multimodal_text_config(monkeypatch, capsys):
    """Nested multimodal text shape fields are copied into draft_hf_config."""
    text_config = SimpleNamespace(
        num_attention_heads=16,
        num_hidden_layers=12,
        hidden_size=2048,
        num_key_value_heads=4,
        head_dim=128,
    )
    config = SimpleNamespace(
        text_config=text_config,
        vision_config=SimpleNamespace(),
    )

    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run", "--", "--enforce-eager"],
        config=config,
    )

    hf_config = _extract_speculative_config(output)["draft_model_config"][
        "hf_config"
    ]
    assert hf_config == {
        "eagle_aux_hidden_state_layer_ids": [2, 6, 9, 12],
        "text_config": None,
        "num_attention_heads": 16,
        "num_hidden_layers": 12,
        "hidden_size": 2048,
        "num_key_value_heads": 4,
        "head_dim": 128,
    }


def test_launch_vllm_rejects_config_without_hidden_layer_count(monkeypatch, capsys):
    """Malformed configs fail with a clear error instead of AttributeError."""
    with pytest.raises(ValueError, match="num_hidden_layers"):
        _run_launch_vllm_dry_run(
            monkeypatch,
            capsys,
            ["dummy", "--dry-run"],
            config=SimpleNamespace(text_config=None),
        )


@pytest.mark.parametrize(
    "include_flag",
    [[], ["--no-include-last-layer"]],
    ids=["auto_include", "explicit_final_only"],
)
def test_launch_vllm_moves_explicit_final_layer_to_unique_last_position(
    monkeypatch,
    capsys,
    include_flag,
):
    """The data loader relies on the final extracted tensor being the verifier final."""
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        [
            "dummy",
            "--dry-run",
            "--target-layer-ids",
            "8",
            "2",
            "4",
            *include_flag,
        ],
        config=SimpleNamespace(num_hidden_layers=8),
    )

    hf_config = _extract_speculative_config(output)["draft_model_config"]["hf_config"]
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [2, 4, 8]


def test_launch_vllm_rejects_duplicate_custom_target_layer_ids(monkeypatch, capsys):
    with pytest.raises(ValueError, match="must not contain duplicate"):
        _run_launch_vllm_dry_run(
            monkeypatch,
            capsys,
            [
                "dummy",
                "--dry-run",
                "--target-layer-ids",
                "2",
                "8",
                "8",
            ],
            config=SimpleNamespace(num_hidden_layers=8),
        )


@pytest.mark.parametrize("layer_id", [0, -1, 9])
def test_launch_vllm_rejects_out_of_range_custom_target_layer_ids(
    monkeypatch,
    capsys,
    layer_id,
):
    with pytest.raises(ValueError, match="inclusive range"):
        _run_launch_vllm_dry_run(
            monkeypatch,
            capsys,
            ["dummy", "--dry-run", "--target-layer-ids", str(layer_id), "2"],
            config=SimpleNamespace(num_hidden_layers=8),
        )


def test_launch_vllm_accepts_final_layer_only_for_mtp(monkeypatch, capsys):
    """MTP legitimately extracts only the verifier's final hidden layer."""
    output = _run_launch_vllm_dry_run(
        monkeypatch,
        capsys,
        ["dummy", "--dry-run", "--target-layer-ids", "8"],
        config=SimpleNamespace(num_hidden_layers=8),
    )

    hf_config = _extract_speculative_config(output)["draft_model_config"]["hf_config"]
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [8]


def test_launch_vllm_no_include_requires_explicit_final_layer(monkeypatch, capsys):
    with pytest.raises(ValueError, match="requires the final verifier layer"):
        _run_launch_vllm_dry_run(
            monkeypatch,
            capsys,
            [
                "dummy",
                "--dry-run",
                "--target-layer-ids",
                "2",
                "4",
                "--no-include-last-layer",
            ],
            config=SimpleNamespace(num_hidden_layers=8),
        )
