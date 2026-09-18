import pytest
import torch

from speculators.models.xpress.model_definitions import XPressRefinerHead


def _head(block_size: int = 4, rank: int = 8) -> XPressRefinerHead:
    torch.manual_seed(0)
    head = XPressRefinerHead(
        verifier_vocab_size=32,
        draft_vocab_size=32,
        hidden_size=16,
        block_size=block_size,
        rank=rank,
        mlp_ratio=2,
    )
    with torch.no_grad():
        head.mix_l.normal_(std=0.1)
    return head


def _inputs(head: XPressRefinerHead, num_blocks: int = 3):
    torch.manual_seed(1)
    hidden = torch.randn(num_blocks, head.block_size, 16)
    prev = torch.randint(0, 32, (num_blocks, head.block_size))
    return hidden, prev


def test_mixer_is_causal_across_the_block():
    """Jacobi iteration is only valid if a settled prefix cannot be disturbed.

    Changing the previous-token id at slot j must leave every slot k < j
    untouched; if the mixer leaked backwards, an earlier slot could flip after it
    had already been accepted and the fixed point would not exist.
    """
    head = _head()
    hidden, prev = _inputs(head)

    with torch.no_grad():
        base = head.block_bias(prev, hidden_states=hidden)

    for j in range(head.block_size):
        perturbed = prev.clone()
        perturbed[:, j] = (perturbed[:, j] + 7) % 32
        with torch.no_grad():
            out = head.block_bias(perturbed, hidden_states=hidden)
        if j > 0:
            torch.testing.assert_close(out[:, :j], base[:, :j])
        assert not torch.allclose(out[:, j], base[:, j]), (
            f"slot {j} did not react to its own input"
        )


def test_hidden_cache_matches_the_full_path():
    """hcache is pass-invariant, so reusing it across Jacobi passes must be exact.

    Training computes it once per block; if it drifted from the full path, the
    refine passes after the first would be trained on different features than
    they see.
    """
    head = _head()
    hidden, prev = _inputs(head)

    with torch.no_grad():
        full = head.block_bias(prev, hidden_states=hidden)
        cached = head.block_bias(prev, hcache=head.hidden_cache(hidden))

    torch.testing.assert_close(full, cached)


def test_fold_matches_the_serving_time_mixer():
    """Serving folds the mixer to ``L * tril + I`` and drops the residual add.

    Training stores raw ``mix_l`` and computes ``x + (L * tril) x``. The two must
    be the same function or an exported checkpoint would mean something different
    at inference time than it did during training.
    """
    head = _head()
    r, b = head.rank, head.block_size
    x = torch.randn(3, b, r)

    l_eff = head.mix_l * head.causal_tril.to(head.mix_l.dtype)
    training = x + torch.einsum("ckj,njc->nkc", l_eff, x)

    folded = l_eff + torch.eye(b).expand(r, -1, -1)
    serving = torch.bmm(folded, x.permute(2, 1, 0)).permute(2, 1, 0)

    torch.testing.assert_close(training, serving)


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank must be > 0"):
        XPressRefinerHead(
            verifier_vocab_size=32,
            draft_vocab_size=32,
            hidden_size=16,
            block_size=4,
            rank=0,
        )
