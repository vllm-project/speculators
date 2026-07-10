from types import SimpleNamespace

from speculators.train.vocab_mapping import get_target_vocab_size


def test_get_target_vocab_size_falls_back_when_text_config_is_none(monkeypatch):
    verifier = SimpleNamespace(text_config=None, vocab_size=321)
    monkeypatch.setattr(
        "speculators.models.utils.AutoConfig.from_pretrained",
        lambda _path: verifier,
    )

    assert get_target_vocab_size(None, "unused-verifier") == 321


def test_explicit_target_vocab_size_does_not_load_model_config(monkeypatch):
    def fail_if_called(_path):
        raise AssertionError("explicit vocab size must not load a model config")

    monkeypatch.setattr(
        "speculators.models.utils.AutoConfig.from_pretrained",
        fail_if_called,
    )

    assert get_target_vocab_size(123, None) == 123
