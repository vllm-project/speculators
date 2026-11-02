import pytest

from speculators.train import utils
from speculators.train.utils import resolve_mask_token_id


class _NoMaskAttrTokenizer:
    """Mimics MistralCommonBackend: reading mask_token_id raises AttributeError."""

    pad_token_id: int | None = 11
    eos_token_id: int | None = 2
    unk_token_id: int | None = 0

    def __len__(self):
        return 131072

    def add_special_tokens(self, tokens):
        raise NotImplementedError("add_special_tokens is not implemented")

    def __getattr__(self, name):
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")


class _CannotAddTokenizer(_NoMaskAttrTokenizer):
    """Defines mask_token_id but, like MistralCommonBackend, can't add tokens."""

    mask_token_id = None


class _HFTokenizer:
    def __init__(self, mask_token_id=None, size=8, pad_token_id=None):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = 1
        self.unk_token_id = None
        self._size = size

    def __len__(self):
        return self._size

    def add_special_tokens(self, tokens):
        self.mask_token_id = self._size
        self._size += 1


@pytest.fixture
def use_tokenizer(monkeypatch):
    def _use(tokenizer):
        monkeypatch.setattr(utils, "load_processor", lambda *_a, **_k: tokenizer)
        monkeypatch.setattr(utils, "get_tokenizer", lambda processor: processor)

    return _use


def test_tokenizer_without_mask_attr_falls_back_to_pad(use_tokenizer):
    use_tokenizer(_NoMaskAttrTokenizer())
    with pytest.warns(UserWarning, match="pad_token_id=11"):
        assert resolve_mask_token_id("mistral", vocab_size=131072) == 11


def test_tokenizer_that_cannot_add_tokens_falls_back_to_pad(use_tokenizer):
    # Embedding padded beyond the tokenizer, but the backend can't add <|MASK|>.
    use_tokenizer(_CannotAddTokenizer())
    with pytest.warns(UserWarning, match="can't add tokens.*pad_token_id=11"):
        assert resolve_mask_token_id("mistral", vocab_size=131200) == 11


def test_explicit_mask_token_id_wins(use_tokenizer):
    use_tokenizer(_NoMaskAttrTokenizer())
    assert resolve_mask_token_id("mistral", vocab_size=131072, mask_token_id=30) == 30


def test_tokenizer_mask_token_id_used(use_tokenizer):
    use_tokenizer(_HFTokenizer(mask_token_id=5))
    assert resolve_mask_token_id("hf", vocab_size=8) == 5


def test_unused_embedding_slot_gets_added_mask(use_tokenizer):
    use_tokenizer(_HFTokenizer(size=8))
    assert resolve_mask_token_id("hf", vocab_size=16) == 8


def test_full_vocab_falls_back_to_eos(use_tokenizer):
    use_tokenizer(_HFTokenizer(size=8))
    with pytest.warns(UserWarning, match="eos_token_id=1"):
        assert resolve_mask_token_id("hf", vocab_size=8) == 1


def test_no_fallback_token_raises(use_tokenizer):
    tokenizer = _CannotAddTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id = tokenizer.unk_token_id = None
    use_tokenizer(tokenizer)
    with pytest.raises(ValueError, match="can't add tokens"):
        resolve_mask_token_id("mistral", vocab_size=131200)
