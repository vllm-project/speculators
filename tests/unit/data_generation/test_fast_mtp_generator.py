"""Unit tests for fast_mtp_generator — generate_and_save_fast_mtp and helpers."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# conftest.py stubs out vllm so this import works without a vllm install
from speculators.data_generation.fast_mtp_generator import (
    _last_hidden_layer,
    _resolve_loss_mask,
    generate_and_save_fast_mtp,
)

H = 64
SEQ_LEN = 10

# Patch the shared utility — both fast_mtp_generator and vllm_hidden_states_generator
# delegate to _model_utils.num_hidden_layers, so all tests patch there.
_NUM_LAYERS_PATH = "speculators.data_generation._model_utils.AutoConfig.from_pretrained"
_VLLM_GEN_PATH = (
    "speculators.data_generation.fast_mtp_generator.VllmHiddenStatesGenerator"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_vllm_item(seq_len: int = SEQ_LEN, hidden_size: int = H) -> dict:
    """Synthetic item matching VllmHiddenStatesGenerator.generate() output."""
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "hidden_states": [torch.randn(seq_len, hidden_size)],
        "loss_mask": None,
    }


def _make_config(num_hidden_layers: int = 28) -> MagicMock:
    cfg = MagicMock()
    cfg.num_hidden_layers = num_hidden_layers
    return cfg


@pytest.fixture
def mock_vllm_cls() -> MagicMock:
    """Patched VllmHiddenStatesGenerator class; instance.generate returns one item."""
    inst = MagicMock()
    inst.generate.return_value = [_make_vllm_item()]
    return MagicMock(return_value=inst)


@pytest.fixture
def saved_dir(mock_vllm_cls: MagicMock, tmp_path: Path) -> Path:
    """Output dir after one generate_and_save_fast_mtp call (seq_len=6, no mask)."""
    mock_vllm_cls.return_value.generate.return_value = [_make_vllm_item(seq_len=6)]
    with (
        patch(_NUM_LAYERS_PATH, return_value=_make_config()),
        patch(_VLLM_GEN_PATH, mock_vllm_cls),
    ):
        generate_and_save_fast_mtp("fake/model", [[0] * 6], tmp_path)
    return tmp_path


# ── _last_hidden_layer ────────────────────────────────────────────────────────


class TestLastHiddenLayer:
    def test_direct_attribute(self) -> None:
        with patch(_NUM_LAYERS_PATH, return_value=_make_config(num_hidden_layers=28)):
            assert _last_hidden_layer("m") == 27

    def test_text_config_fallback(self) -> None:
        """Multimodal configs nest num_hidden_layers inside text_config."""
        cfg = MagicMock(spec=[])  # no num_hidden_layers attribute
        cfg.text_config = MagicMock()
        cfg.text_config.num_hidden_layers = 32
        with patch(_NUM_LAYERS_PATH, return_value=cfg):
            assert _last_hidden_layer("m") == 31

    def test_missing_raises(self) -> None:
        with (
            patch(_NUM_LAYERS_PATH, return_value=MagicMock(spec=[])),
            pytest.raises(ValueError, match="num_hidden_layers"),
        ):
            _last_hidden_layer("bad/model")


# ── _resolve_loss_mask ────────────────────────────────────────────────────────


class TestResolveLossMask:
    def _ids(self, n: int = 4) -> torch.Tensor:
        return torch.arange(n, dtype=torch.long)

    def test_fn_takes_priority_over_all(self) -> None:
        """loss_mask_fn wins even when precomputed and raw are both present."""
        raw = torch.zeros(4, dtype=torch.long)
        result = _resolve_loss_mask(
            self._ids(),
            raw,
            precomputed=[0, 0, 0, 0],
            loss_mask_fn=lambda _: [1, 1, 1, 1],
        )
        assert result.tolist() == [1, 1, 1, 1]

    def test_precomputed_used_when_no_fn(self) -> None:
        result = _resolve_loss_mask(
            self._ids(), raw_loss_mask=None, precomputed=[0, 1, 0, 1], loss_mask_fn=None
        )
        assert result.tolist() == [0, 1, 0, 1]

    def test_raw_used_when_no_fn_or_precomputed(self) -> None:
        raw = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        result = _resolve_loss_mask(
            self._ids(), raw, precomputed=None, loss_mask_fn=None
        )
        assert torch.equal(result, raw)

    def test_ones_fallback_when_all_none(self) -> None:
        result = _resolve_loss_mask(
            self._ids(6), None, precomputed=None, loss_mask_fn=None
        )
        assert result.tolist() == [1, 1, 1, 1, 1, 1]

    def test_fn_receives_token_ids(self) -> None:
        """fn is called with the decoded token ID list, not the raw tensor."""
        received: list[list[int]] = []

        def capturing_fn(x: list[int]) -> list[int]:
            received.append(x)
            return [1, 1, 1]

        ids = torch.tensor([10, 20, 30], dtype=torch.long)
        _resolve_loss_mask(ids, None, precomputed=None, loss_mask_fn=capturing_fn)
        assert received == [[10, 20, 30]]


# ── generate_and_save_fast_mtp ────────────────────────────────────────────────


class TestGenerateAndSave:
    def _run(
        self,
        mock_vllm_cls: MagicMock,
        tmp_path: Path,
        items: list[dict] | None = None,
        loss_masks: list[list[int]] | None = None,
        loss_mask_fn: Callable[[list[int]], list[int]] | None = None,
    ) -> None:
        if items is not None:
            mock_vllm_cls.return_value.generate.return_value = items
        with (
            patch(_NUM_LAYERS_PATH, return_value=_make_config()),
            patch(_VLLM_GEN_PATH, mock_vllm_cls),
        ):
            generate_and_save_fast_mtp(
                "fake/model",
                [[0] * 6],
                tmp_path,
                loss_masks=loss_masks,
                loss_mask_fn=loss_mask_fn,
            )

    def test_vllm_receives_last_layer_id(
        self, mock_vllm_cls: MagicMock, tmp_path: Path
    ) -> None:
        """_last_hidden_layer result flows directly to the layer_ids vLLM receives."""
        self._run(mock_vllm_cls, tmp_path)
        assert mock_vllm_cls.call_args.kwargs["layer_ids"] == [27]

    def test_pt_files_written(self, mock_vllm_cls: MagicMock, tmp_path: Path) -> None:
        """One file per sequence, no extras."""
        items = [_make_vllm_item(seq_len=6), _make_vllm_item(seq_len=8)]
        self._run(mock_vllm_cls, tmp_path, items=items)
        assert (tmp_path / "data_0.pt").exists()
        assert (tmp_path / "data_1.pt").exists()
        assert not (tmp_path / "data_2.pt").exists()

    def test_saved_file_contract(self, saved_dir: Path) -> None:
        """Saved .pt has the three keys consumers expect, with correct shapes."""
        saved = torch.load(str(saved_dir / "data_0.pt"), weights_only=True)
        assert set(saved.keys()) == {"input_ids", "hidden_states", "loss_mask"}
        assert saved["input_ids"].shape == (6,)
        assert saved["hidden_states"].shape == (6, H)
        assert saved["loss_mask"].shape == (6,)

    def test_sample_lengths_json(self, saved_dir: Path) -> None:
        lengths = json.loads((saved_dir / "sample_lengths.json").read_text())
        assert lengths == {"0": 6}

    def test_precomputed_loss_masks_flow_to_file(
        self, mock_vllm_cls: MagicMock, tmp_path: Path
    ) -> None:
        """loss_masks= parameter survives the full pipeline to the saved tensor."""
        item = {
            "input_ids": torch.tensor(list(range(6)), dtype=torch.long),
            "hidden_states": [torch.randn(6, H)],
            "loss_mask": None,
        }
        self._run(
            mock_vllm_cls, tmp_path, items=[item], loss_masks=[[0, 0, 0, 1, 1, 1]]
        )
        saved = torch.load(str(tmp_path / "data_0.pt"), weights_only=True)
        assert saved["loss_mask"].tolist() == [0, 0, 0, 1, 1, 1]

    def test_custom_loss_mask_fn_flow_to_file(
        self, mock_vllm_cls: MagicMock, tmp_path: Path
    ) -> None:
        """loss_mask_fn= parameter survives the full pipeline to the saved tensor."""
        item = {
            "input_ids": torch.tensor(list(range(6)), dtype=torch.long),
            "hidden_states": [torch.randn(6, H)],
            "loss_mask": None,
        }
        self._run(
            mock_vllm_cls,
            tmp_path,
            items=[item],
            loss_mask_fn=lambda _ids: [0, 0, 0, 1, 1, 1],
        )
        saved = torch.load(str(tmp_path / "data_0.pt"), weights_only=True)
        assert saved["loss_mask"].tolist() == [0, 0, 0, 1, 1, 1]

    def test_raises_if_both_loss_masks_and_fn(self, tmp_path: Path) -> None:
        # Validation fires before the generator is created — no patches needed.
        with pytest.raises(ValueError, match="not both"):
            generate_and_save_fast_mtp(
                "fake/model",
                [[0] * 6],
                tmp_path,
                loss_masks=[[1] * 6],
                loss_mask_fn=lambda _: [1] * 6,
            )

    def test_raises_if_loss_masks_length_mismatch(self, tmp_path: Path) -> None:
        # Validation fires before the generator is created — no patches needed.
        with pytest.raises(ValueError, match="length"):
            generate_and_save_fast_mtp(
                "fake/model",
                [[0] * 6],  # 1 sequence
                tmp_path,
                loss_masks=[[1] * 6, [1] * 6],  # 2 masks — mismatch
            )

    def test_hidden_states_unwrapped_from_list(
        self, mock_vllm_cls: MagicMock, tmp_path: Path
    ) -> None:
        """VllmHiddenStatesGenerator returns list[Tensor]; we extract element [0]."""
        hs = torch.randn(6, H)
        item = {
            "input_ids": torch.arange(6, dtype=torch.long),
            "hidden_states": [hs],
            "loss_mask": None,
        }
        self._run(mock_vllm_cls, tmp_path, items=[item])
        saved = torch.load(str(tmp_path / "data_0.pt"), weights_only=True)
        assert torch.equal(saved["hidden_states"], hs)

    def test_hidden_states_passthrough_when_tensor(
        self, mock_vllm_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Plain tensor hidden_states are accepted without unwrapping."""
        hs = torch.randn(6, H)
        item = {
            "input_ids": torch.arange(6, dtype=torch.long),
            "hidden_states": hs,
            "loss_mask": None,
        }
        self._run(mock_vllm_cls, tmp_path, items=[item])
        saved = torch.load(str(tmp_path / "data_0.pt"), weights_only=True)
        assert torch.equal(saved["hidden_states"], hs)
