"""Prepare offline DSpark calibration inputs and run its loss-free backbone."""

from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from speculators.data_generation.offline import check_hidden_states
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.models.dspark import DSparkDraftModel


class CalibrationDataset(Dataset):
    """Provide validated DSpark calibration samples."""

    def __init__(
        self,
        records: HFDataset,
        hidden_states_dir: Path,
        *,
        count: int,
        max_length: int,
        max_anchors: int,
        layer_ids: Sequence[int],
        width: int,
        block_size: int,
    ) -> None:
        """Initialize the dataset and validate its sampling limits.

        Args:
            records: Token records loaded from disk.
            hidden_states_dir: Directory containing per-sample safetensors files.
            count: Number of records to expose.
            max_length: Maximum input sequence length.
            max_anchors: Maximum anchored draft blocks per sample.
            layer_ids: Target hidden-state layer IDs.
            width: Width of each target hidden state.
            block_size: Number of draft positions per anchored block.
        """
        if not 0 < count <= len(records):
            raise ValueError(f"count must be in [1, {len(records)}], got {count}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if max_length <= block_size:
            raise ValueError(
                f"max_length must exceed block_size ({block_size}), got {max_length}"
            )
        if max_anchors <= 0:
            raise ValueError(f"max_anchors must be positive, got {max_anchors}")
        if not layer_ids:
            raise ValueError("layer_ids must not be empty")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.records = records
        self.hidden_states_dir = hidden_states_dir
        self.count = count
        self.max_length = max_length
        self.max_anchors = max_anchors
        self.block_size = block_size
        self.layers = len(layer_ids)
        self.width = width

    def __len__(self) -> int:
        """Return the number of calibration samples."""
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        """Load and validate one calibration sample.

        Args:
            index: Prepared record and hidden-state file index.

        Returns:
            Single-sample model inputs and the number of anchored blocks.
        """
        record = self.records[index]
        data = load_file(
            filename=str(self.hidden_states_dir / f"hs_{index}.safetensors")
        )
        hs = data.get("hidden_states")
        if hs is None:
            raise ValueError(f"Missing hidden states for sample {index}")
        if hs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported hidden-state dtype {hs.dtype}")
        if (
            hs.ndim != 3
            or hs.shape[1] not in (self.layers, self.layers + 1)
            or hs.shape[2] != self.width
        ):
            raise ValueError(
                f"Invalid hidden-state shape for sample {index}: {hs.shape}"
            )
        check_hidden_states(data=data, tokens=record["input_ids"].tolist())
        length = min(len(record["input_ids"]), self.max_length)
        if record["loss_mask"].ndim != 1 or len(record["loss_mask"]) != len(
            record["input_ids"]
        ):
            raise ValueError(f"Invalid loss mask for sample {index}")
        loss_mask = record["loss_mask"][:length]
        anchors = min(self.max_anchors, int(loss_mask[: -self.block_size].bool().sum()))
        if anchors == 0:
            raise ValueError(f"No valid anchors for sample {index}")
        return {
            "hidden_states": hs[:length, : self.layers]
            .flatten(start_dim=1)
            .to(dtype=torch.bfloat16)
            .unsqueeze(dim=0),
            "input_ids": record["input_ids"][:length].unsqueeze(dim=0),
            "loss_mask": loss_mask.unsqueeze(dim=0),
            "document_ids": torch.zeros(size=(1, length), dtype=torch.long),
            "position_ids": torch.arange(end=length).unsqueeze(dim=0),
            "max_anchors": anchors,
        }


def calibration_batches(
    data_dir: str | Path,
    hidden_states_dir: str | Path,
    *,
    count: int,
    max_length: int,
    max_anchors: int,
    layer_ids: Sequence[int],
    width: int,
    block_size: int,
) -> DataLoader:
    """Create a loader over offline DSpark calibration samples.

    Args:
        data_dir: Hugging Face dataset directory.
        hidden_states_dir: Directory containing per-sample safetensors files.
        count: Number of records to expose.
        max_length: Maximum input sequence length.
        max_anchors: Maximum anchored draft blocks per sample.
        layer_ids: Target hidden-state layer IDs.
        width: Width of each target hidden state.
        block_size: Number of draft positions per anchored block.

    Returns:
        Data loader that yields individual calibration samples without collation.
    """
    records = load_from_disk(dataset_path=str(data_dir)).with_format(type="torch")
    dataset = CalibrationDataset(
        records=records,
        hidden_states_dir=Path(hidden_states_dir),
        count=count,
        max_length=max_length,
        max_anchors=max_anchors,
        layer_ids=layer_ids,
        width=width,
        block_size=block_size,
    )
    return DataLoader(dataset=dataset, batch_size=None, num_workers=0)


def calibration_forward(
    model: DSparkDraftModel,
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    position_ids: torch.Tensor,
    max_anchors: int,
) -> torch.Tensor:
    """Run the backbone without training's verifier targets, logits, or losses.

    Calibration only needs activations, and the compressor may disable lm_head.

    Args:
        model: DSpark model to run.
        hidden_states: Target hidden states.
        input_ids: Input token IDs.
        loss_mask: Token loss mask used to select anchors.
        document_ids: Document boundaries for attention.
        position_ids: Token position IDs.
        max_anchors: Maximum number of anchored blocks.

    Returns:
        Normalized backbone activations for the anchored draft blocks.
    """
    device = hidden_states.device
    full_mask, sliding_mask, anchors, valid = model._build_attention_mask(  # noqa: SLF001
        loss_mask=loss_mask,
        max_anchors=max_anchors,
        document_ids=document_ids,
        device=device,
    )
    if not bool(valid.all()):
        raise ValueError("Calibration must not include padded anchors")
    mask_ids = torch.full(
        size=(1, max_anchors * model.block_size),
        fill_value=model.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    mask_ids[:, :: model.block_size] = input_ids[:, anchors]
    noise = model.embed_tokens(input=mask_ids)
    target = model.hidden_norm(hidden_states=model.fc(input=hidden_states))
    positions = get_base_indices_for_anchored_blocks(
        anchor_positions=position_ids[0, anchors], block_size=model.block_size
    )
    position_ids = torch.cat(tensors=[position_ids, positions.unsqueeze(dim=0)], dim=1)
    embeddings = model.rotary_emb(x=hidden_states, position_ids=position_ids)
    for idx, layer in enumerate(model.layers):
        noise = layer(
            hidden_states=noise,
            target_hidden=target,
            attention_mask=sliding_mask
            if idx in model.sliding_window_indices
            else full_mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=embeddings,
        )
    return model.norm(hidden_states=noise)
