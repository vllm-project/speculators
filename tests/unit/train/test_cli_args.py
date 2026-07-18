"""Tests for CLI arguments."""

import pytest

from scripts.train import parse_args
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import ce_loss, kl_div_loss, tv_loss
from speculators.models.peagle.core import PEagleDraftModel


def _parse(monkeypatch, extra: list[str]):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "dummy"] + extra
    )
    return parse_args()


def test_shared_hidden_state_cache_is_opt_in(monkeypatch):
    args = _parse(monkeypatch, [])

    assert args.shared_hidden_states_path is None
    assert args.shared_hidden_states_namespace is None
    assert args.shared_hidden_states_ttl == 3600.0
    assert args.shared_hidden_states_lock_timeout == 300.0
    assert args.shared_hidden_states_consumer_id is None
    assert args.shared_hidden_states_lookbehind == 2
    assert args.shared_hidden_states_lookahead == 40
    assert args.shared_hidden_states_max_prefetch_per_consumer == 8
    assert args.shared_hidden_states_capture_batch_size == 8
    assert args.shared_hidden_states_capture_batch_wait == 0.002
    assert args.shared_hidden_states_max_inflight == 32


def test_shared_hidden_state_cache_arguments(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--shared-hidden-states-path",
            "shared-cache",
            "--shared-hidden-states-namespace",
            "layers:2,18,33",
            "--shared-hidden-states-ttl",
            "0",
            "--shared-hidden-states-lock-timeout",
            "45",
        ],
    )

    assert args.shared_hidden_states_path == "shared-cache"
    assert args.shared_hidden_states_namespace == "layers:2,18,33"
    assert args.shared_hidden_states_ttl == 0
    assert args.shared_hidden_states_lock_timeout == 45


def test_windowed_shared_hidden_state_arguments(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--shared-hidden-states-path",
            "shared-cache",
            "--shared-hidden-states-consumer-id",
            "consumer-a",
            "--shared-hidden-states-lookbehind",
            "3",
            "--shared-hidden-states-lookahead",
            "20",
            "--shared-hidden-states-max-prefetch-per-consumer",
            "7",
            "--shared-hidden-states-capture-batch-size",
            "6",
            "--shared-hidden-states-capture-batch-wait",
            "0.01",
            "--shared-hidden-states-max-inflight",
            "40",
            "--shared-hidden-states-consumer-timeout",
            "60",
            "--shared-hidden-states-claim-timeout",
            "90",
            "--shared-hidden-states-generation-attempts",
            "4",
        ],
    )

    assert args.shared_hidden_states_consumer_id == "consumer-a"
    assert args.shared_hidden_states_lookbehind == 3
    assert args.shared_hidden_states_lookahead == 20
    assert args.shared_hidden_states_max_prefetch_per_consumer == 7
    assert args.shared_hidden_states_capture_batch_size == 6
    assert args.shared_hidden_states_capture_batch_wait == 0.01
    assert args.shared_hidden_states_max_inflight == 40
    assert args.shared_hidden_states_consumer_timeout == 60
    assert args.shared_hidden_states_claim_timeout == 90
    assert args.shared_hidden_states_generation_attempts == 4


@pytest.mark.parametrize(
    "extra",
    [
        ["--shared-hidden-states-path", ""],
        ["--shared-hidden-states-namespace", "namespace-only"],
        ["--shared-hidden-states-path", "cache", "--legacy-data"],
        [
            "--shared-hidden-states-path",
            "cache",
            "--shared-hidden-states-namespace",
            "",
        ],
        ["--shared-hidden-states-ttl", "-1"],
        ["--shared-hidden-states-lock-timeout", "0"],
        ["--shared-hidden-states-consumer-id", "consumer"],
        [
            "--shared-hidden-states-path",
            "cache",
            "--shared-hidden-states-consumer-id",
            "consumer",
            "--on-missing",
            "raise",
        ],
        [
            "--shared-hidden-states-path",
            "cache",
            "--shared-hidden-states-consumer-id",
            "",
        ],
        ["--shared-hidden-states-lookbehind", "-1"],
        ["--shared-hidden-states-lookahead", "-1"],
        ["--shared-hidden-states-max-prefetch-per-consumer", "-1"],
        [
            "--shared-hidden-states-lookahead",
            "0",
            "--shared-hidden-states-max-prefetch-per-consumer",
            "2",
        ],
        ["--shared-hidden-states-capture-batch-size", "0"],
        ["--shared-hidden-states-capture-batch-wait", "-0.001"],
        ["--shared-hidden-states-max-inflight", "0"],
        ["--shared-hidden-states-consumer-timeout", "0"],
        ["--shared-hidden-states-claim-timeout", "0"],
        ["--shared-hidden-states-generation-attempts", "0"],
    ],
)
def test_shared_hidden_state_cache_rejects_invalid_combinations(monkeypatch, extra):
    with pytest.raises(SystemExit, match="2"):
        _parse(monkeypatch, extra)


# ---------------------------------------------------------------------------
# Ensure CLI args flow correctly through vars(args) into get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_dflash_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_decay_gamma(monkeypatch):
    args = _parse(monkeypatch, ["--dflash-decay-gamma", "7.0"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["gamma"] == 7.0
    assert val_kw["gamma"] == 7.0


def test_dflash_decay_gamma_falls_back_when_omitted():
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(loss_fn="kl_div")
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_eagle3_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_eagle3_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_peagle_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_peagle_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_dspark_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["confidence_head_alpha"] == 1.0
    assert val_kw["confidence_head_alpha"] == 1.0


def test_dspark_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["tv"][0] is tv_loss
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_dspark_confidence_head_alpha(monkeypatch):
    args = _parse(monkeypatch, ["--confidence-head-alpha", "0.5"])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["confidence_head_alpha"] == 0.5
    assert val_kw["confidence_head_alpha"] == 0.5


# ---------------------------------------------------------------------------
# Per-speculator-type defaults for draft_arch, norm_before_fc, norm_output
# ---------------------------------------------------------------------------


def test_eagle3_defaults_to_llama_arch(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.draft_arch == "llama"


def test_eagle3_defaults_norm_before_fc_true(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.norm_before_fc is True


def test_eagle3_defaults_norm_output_true(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.norm_output is True


def test_dflash_defaults_to_qwen3_arch(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.draft_arch == "qwen3"


def test_dflash_defaults_norm_before_fc_false(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.norm_before_fc is False


def test_dflash_defaults_norm_output_false(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.norm_output is False


def test_no_norm_before_fc_flag(monkeypatch):
    args = _parse(monkeypatch, ["--no-norm-before-fc"])
    assert args.norm_before_fc is False


def test_no_norm_output_flag(monkeypatch):
    args = _parse(monkeypatch, ["--no-norm-output"])
    assert args.norm_output is False


def test_consumer_optimization_defaults_preserve_existing_backends(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])

    assert args.dflash_linear_cross_entropy_backend == "torch"
    assert not args.dflash_compact_zero_weight_ce_rows
    assert args.dflash_label_source == "verifier_argmax"
    assert args.dflash_verifier_argmax_chunk_size == 0
    assert args.adamw_backend == "auto"
    assert args.gradient_clip_backend == "torch"
    assert args.max_grad_norm == 1.0


def test_consumer_optimization_arguments_flow_to_dflash(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dflash",
            "--loss-fn",
            "ce",
            "--dflash-linear-cross-entropy-backend",
            "liger",
            "--dflash-compact-zero-weight-ce-rows",
            "--dflash-label-source",
            "input_ids",
            "--dflash-verifier-argmax-chunk-size",
            "512",
            "--optimizer",
            "adamw",
            "--adamw-backend",
            "fused",
            "--gradient-clip-backend",
            "fused_adamw",
            "--max-grad-norm",
            "0.75",
        ],
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "speculators.models.dflash.core.validate_liger_installation", lambda: None
        )
        train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))

    assert train_kw["linear_cross_entropy_backend"] == "liger"
    assert train_kw["compact_zero_weight_ce_rows"] is True
    assert train_kw["label_source"] == "input_ids"
    assert train_kw["verifier_argmax_chunk_size"] == 512
    assert val_kw == train_kw
    assert args.adamw_backend == "fused"
    assert args.gradient_clip_backend == "fused_adamw"
    assert args.max_grad_norm == 0.75


@pytest.mark.parametrize(
    "extra",
    [
        ["--speculator-type", "dflash", "--dflash-compact-zero-weight-ce-rows"],
        ["--speculator-type", "dflash", "--dflash-label-source", "input_ids"],
        ["--speculator-type", "dflash", "--dflash-verifier-argmax-chunk-size", "-1"],
        ["--gradient-clip-backend", "fused_adamw"],
    ],
)
def test_consumer_optimization_rejects_invalid_combinations(monkeypatch, extra):
    with pytest.raises(SystemExit, match="2"):
        _parse(monkeypatch, extra)
