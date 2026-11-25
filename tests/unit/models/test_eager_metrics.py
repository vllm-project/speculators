import pytest
import torch

from speculators.losses import compound_loss, eager, resolve_loss_config
from speculators.losses.utils import _EAGER_LOSS_FN_MAP


@pytest.mark.parametrize("chunk_rows", [None, 8])
def test_compound_loss_chunks_eager_metrics(monkeypatch, chunk_rows):
    expected_rows = 1024 if chunk_rows is None else chunk_rows
    logits = torch.randn(1, expected_rows + 1, 7, requires_grad=True)
    targets = torch.randn_like(logits)
    mask = torch.ones_like(logits[..., 0])
    positions = torch.arange(logits.shape[1]).unsqueeze(0)
    config = resolve_loss_config("ce", implementation="eager")
    calls = []
    original_tv = eager.tv_loss
    original_nla = eager.neg_log_acceptance_loss

    def record(name, fn):
        def metric(draft, target):
            assert not torch.is_grad_enabled()
            calls.append((name, draft.shape[-2]))
            return fn(draft, target)

        return metric

    monkeypatch.setitem(_EAGER_LOSS_FN_MAP, "tv", record("tv", original_tv))
    monkeypatch.setitem(_EAGER_LOSS_FN_MAP, "nla", record("nla", original_nla))
    baseline = compound_loss(logits, targets, mask, positions, config)
    chunk_kwargs = {} if chunk_rows is None else {"chunk_rows": chunk_rows}
    output = compound_loss(
        logits,
        targets,
        mask,
        positions,
        config,
        metric_terms=("tv", "nla"),
        with_ids=True,
        **chunk_kwargs,
    )

    assert calls == [
        ("tv", expected_rows),
        ("nla", expected_rows),
        ("tv", 1),
        ("nla", 1),
    ]
    with torch.no_grad():
        torch.testing.assert_close(
            output.metric_terms["tv"], original_tv(logits, targets)
        )
        torch.testing.assert_close(
            output.metric_terms["nla"], original_nla(logits, targets)
        )
    torch.testing.assert_close(output.pred_ids, logits.argmax(dim=-1))
    torch.testing.assert_close(output.target_ids, targets.argmax(dim=-1))
    torch.testing.assert_close(output.loss, baseline.loss)
    torch.testing.assert_close(
        torch.autograd.grad(output.loss, logits),
        torch.autograd.grad(baseline.loss, logits),
    )
