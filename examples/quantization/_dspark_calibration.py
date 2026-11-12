"""Offline DSpark calibration inputs and a loss-free backbone forward."""

import json
import shlex
from pathlib import Path

import torch
from hs_connectors.transfer import FileTransfer
from torch.utils.data import DataLoader, Dataset

from speculators.data_generation.offline import check_hidden_states
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.train.data import ArrowDataset
from speculators.train.hidden_states_source import OfflineHiddenStatesSource
from speculators.train.recovery import SampleUnavailable


def capture_layers(provenance: Path, verifier: str) -> list[int]:
    lines = [
        line for line in provenance.read_text().splitlines() if not line.startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError("Expected one vLLM command in extraction provenance")
    command = shlex.split(lines[0])
    if command[command.index("serve") + 1] != verifier:
        raise ValueError("Extraction verifier does not match drafter verifier")
    config = json.loads(command[command.index("--speculative_config") + 1])
    if config["method"] != "extract_hidden_states":
        raise ValueError("Provenance does not describe hidden-state extraction")
    return config["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"]


class CalibrationDataset(Dataset):
    def __init__(
        self,
        records,
        source,
        *,
        count,
        max_length,
        max_anchors,
        layer_ids,
        captured_layer_ids,
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
        if captured_layer_ids[: len(layer_ids)] != layer_ids or len(
            captured_layer_ids
        ) not in (len(layer_ids), len(layer_ids) + 1):
            raise ValueError("Captured layer order does not match the drafter")
        self.records, self.source, self.count = records, source, count
        self.max_length, self.max_anchors, self.block_size = (
            max_length,
            max_anchors,
            block_size,
        )
        self.layers, self.captured_layers, self.width = (
            len(layer_ids),
            len(captured_layer_ids),
            width,
        )

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        record = self.records[index]
        data = self.source.get_one(record)
        if isinstance(data, SampleUnavailable):
            raise ValueError(f"Missing hidden states for sample {index}")
        check_hidden_states(data, record.input_ids.tolist())
        hs = data["hidden_states"]
        if hs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported hidden-state dtype {hs.dtype}")
        if hs.ndim != 3 or tuple(hs.shape[1:]) != (self.captured_layers, self.width):
            raise ValueError(
                f"Invalid hidden-state shape for sample {index}: {hs.shape}"
            )
        length = min(len(record.input_ids), self.max_length)
        if record.loss_mask.ndim != 1 or len(record.loss_mask) != len(record.input_ids):
            raise ValueError(f"Invalid loss mask for sample {index}")
        loss_mask = record.loss_mask[:length]
        anchors = min(self.max_anchors, int(loss_mask[: -self.block_size].bool().sum()))
        if anchors == 0:
            raise ValueError(f"No valid anchors for sample {index}")
        return {
            "hidden_states": hs[:length, : self.layers]
            .flatten(1)
            .to(torch.bfloat16)
            .unsqueeze(0),
            "input_ids": record.input_ids[:length].unsqueeze(0),
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
    captured_layer_ids,
    width,
    block_size,
):
    records = ArrowDataset(data_dir)
    source = OfflineHiddenStatesSource(
        transfer=FileTransfer(Path(hidden_states_dir)), on_missing="raise"
    )
    dataset = CalibrationDataset(
        records,
        source,
        count=count,
        max_length=max_length,
        max_anchors=max_anchors,
        layer_ids=layer_ids,
        captured_layer_ids=captured_layer_ids,
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
    """Run the same anchored backbone as training, without logits or losses."""
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
