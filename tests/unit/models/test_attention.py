import pytest
import torch

from speculators.models import attention


def test_flex_attention_forward_returns_transposed_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    flex_output = torch.randn(1, 2, 3, 4)
    mask = object()

    def fake_flex_attention(*args, **kwargs):
        assert args == (query, key, value)
        assert kwargs == {
            "score_mod": None,
            "block_mask": mask,
            "enable_gqa": True,
            "scale": 0.5,
        }
        return flex_output

    monkeypatch.setattr(attention, "flex_attention", fake_flex_attention)

    output, weights = attention.flex_attention_forward(
        torch.nn.Identity(), query, key, value, mask, scaling=0.5
    )

    torch.testing.assert_close(output, flex_output.transpose(1, 2))
    assert output.is_contiguous()
    assert weights is None


def test_flex_attention_forward_rejects_auxiliary_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.randn(1, 1, 2, 4)
    monkeypatch.setattr(
        attention,
        "flex_attention",
        lambda *args, **kwargs: (tensor, torch.zeros(1)),
    )

    with pytest.raises(
        TypeError, match="Flex Attention unexpectedly returned auxiliary output"
    ):
        attention.flex_attention_forward(
            torch.nn.Identity(), tensor, tensor, tensor, attention_mask=None
        )
