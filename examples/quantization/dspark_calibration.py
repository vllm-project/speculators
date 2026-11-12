"""Prepare offline DSpark calibration inputs and run its loss-free backbone."""

from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from speculators.data_generation.offline import check_hidden_states
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks


class CalibrationDataset(Dataset):
    def __init__(
        self,
        records,
        hidden_states_dir,
        *,
        count,
        max_length,
        max_anchors,
        layer_ids,
        width,
        block_size,
    ):
        if (
            count <= 0
            or count > len(records)
            or max_length <= block_size
            or max_anchors <= 0
        ):
            raise ValueError("Invalid sample count, sequence length, or anchor count")
        self.records, self.hidden_states_dir, self.count = (
            records,
            hidden_states_dir,
            count,
        )
        self.max_length, self.max_anchors, self.block_size = (
            max_length,
            max_anchors,
            block_size,
        )
        self.layers, self.width = len(layer_ids), width

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        record = self.records[index]
        data = load_file(str(self.hidden_states_dir / f"hs_{index}.safetensors"))
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
        check_hidden_states(data, record["input_ids"].tolist())
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
            .flatten(1)
            .to(torch.bfloat16)
            .unsqueeze(0),
            "input_ids": record["input_ids"][:length].unsqueeze(0),
            "loss_mask": loss_mask.unsqueeze(0),
            "document_ids": torch.zeros((1, length), dtype=torch.long),
            "position_ids": torch.arange(length).unsqueeze(0),
            "max_anchors": anchors,
        }


def calibration_batches(
    data_dir,
    hidden_states_dir,
    *,
    count,
    max_length,
    max_anchors,
    layer_ids,
    width,
    block_size,
):
    records = load_from_disk(str(data_dir)).with_format("torch")
    dataset = CalibrationDataset(
        records,
        Path(hidden_states_dir),
        count=count,
        max_length=max_length,
        max_anchors=max_anchors,
        layer_ids=layer_ids,
        width=width,
        block_size=block_size,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


def calibration_forward(
    model,
    *,
    hidden_states,
    input_ids,
    loss_mask,
    document_ids,
    position_ids,
    max_anchors,
):
    """Run the backbone without training's verifier targets, logits, or losses.

    Calibration only needs activations, and the compressor may disable lm_head.
    """
    device = hidden_states.device
    full_mask, sliding_mask, anchors, valid = model._build_attention_mask(  # noqa: SLF001
        loss_mask, max_anchors, document_ids, device
    )
    if not bool(valid.all()):
        raise ValueError("Calibration must not include padded anchors")
    mask_ids = torch.full(
        (1, max_anchors * model.block_size),
        model.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    mask_ids[:, :: model.block_size] = input_ids[:, anchors]
    noise = model.embed_tokens(mask_ids)
    target = model.hidden_norm(model.fc(hidden_states))
    positions = get_base_indices_for_anchored_blocks(
        position_ids[0, anchors], model.block_size
    )
    position_ids = torch.cat([position_ids, positions.unsqueeze(0)], dim=1)
    embeddings = model.rotary_emb(hidden_states, position_ids)
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
    return model.norm(noise)
