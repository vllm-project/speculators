"""Tests for CLI arguments."""

from scripts.train import parse_args
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import ce_loss, kl_div_loss
from speculators.models.peagle.core import PEagleDraftModel


def _parse(monkeypatch, extra: list[str]):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "dummy"] + extra
    )
    return parse_args()


# ---------------------------------------------------------------------------
# Ensure CLI args flow correctly through vars(args) into get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_dflash_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is kl_div_loss
    assert val_kw["loss_fn"] is kl_div_loss


def test_dflash_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is ce_loss
    assert val_kw["loss_fn"] is ce_loss


def test_eagle3_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is kl_div_loss
    assert val_kw["loss_fn"] is kl_div_loss


def test_eagle3_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is ce_loss
    assert val_kw["loss_fn"] is ce_loss


def test_peagle_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is kl_div_loss
    assert val_kw["loss_fn"] is kl_div_loss


def test_peagle_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is ce_loss
    assert val_kw["loss_fn"] is ce_loss
