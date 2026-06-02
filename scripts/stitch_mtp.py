#!/usr/bin/env python3
"""
Stitch finetuned MTP weights back into a verifier checkpoint.

Takes a finetuned MTP speculator checkpoint (speculators format) and merges
the trained weights back into the original verifier checkpoint, producing a
self-contained checkpoint directory deployable on vLLM.

Frozen weights (embed_tokens, lm_head) are skipped -- only the MTP layer
weights are replaced.

Usage:
    python scripts/stitch_mtp.py ./finetuned-mtp ./Qwen3-Next-80B-A3B-Instruct

    # verifier can be a HuggingFace model ID:
    python scripts/stitch_mtp.py ./finetuned-mtp Qwen/Qwen3-Next-80B-A3B-Instruct

    # custom output path (defaults to {verifier-name}-stitched):
    python scripts/stitch_mtp.py ./finetuned-mtp ./verifier --output-path ./out
"""

import json
import re
import shutil
from pathlib import Path
from typing import Annotated

import torch
import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from safetensors import safe_open
from safetensors.torch import save_file

from speculators.convert.mtp import MTP_EXACT_REMAP, MTP_PREFIX_REMAP

app = typer.Typer(rich_markup_mode="rich")
console = Console()

INVERSE_MTP_EXACT_REMAP = {v: k for k, v in MTP_EXACT_REMAP.items()}
INVERSE_MTP_PREFIX_REMAP = [(dst, src) for src, dst in MTP_PREFIX_REMAP]

_FROZEN_KEYS = {"embed_tokens.weight", "lm_head.weight"}

_FUSED_GATE_UP_PATTERN = re.compile(r"^(.+\.experts)\.gate_up_proj$")
_FUSED_DOWN_PATTERN = re.compile(r"^(.+\.experts)\.down_proj$")


def _spinner() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )


def _bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} shards"),
        TimeElapsedColumn(),
        console=console,
    )


def _remap_key(key: str) -> str:
    if key in INVERSE_MTP_EXACT_REMAP:
        return INVERSE_MTP_EXACT_REMAP[key]
    for src, dst in INVERSE_MTP_PREFIX_REMAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    return key


def _filter_frozen_keys(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in weights.items() if k not in _FROZEN_KEYS}


def _unfuse_moe_experts(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Unfuse packed 3D expert tensors back to per-expert format.

    Inverse of ``MTPConverter._fuse_moe_experts``.
    """
    result: dict[str, torch.Tensor] = {}
    gate_up_keys: dict[str, torch.Tensor] = {}
    down_keys: dict[str, torch.Tensor] = {}

    for key, tensor in weights.items():
        m_gu = _FUSED_GATE_UP_PATTERN.match(key)
        m_d = _FUSED_DOWN_PATTERN.match(key)
        if m_gu:
            gate_up_keys[m_gu.group(1)] = tensor
        elif m_d:
            down_keys[m_d.group(1)] = tensor
        else:
            result[key] = tensor

    if not gate_up_keys:
        return weights

    for prefix, gate_up in gate_up_keys.items():
        if prefix not in down_keys:
            raise ValueError(f"Found gate_up_proj at '{prefix}' but missing down_proj")
        down = down_keys[prefix]

        num_experts = gate_up.shape[0]
        if down.shape[0] != num_experts:
            raise ValueError(
                f"Expert count mismatch at '{prefix}': "
                f"gate_up_proj has {num_experts} but down_proj has {down.shape[0]}"
            )
        half = gate_up.shape[1] // 2

        for i in range(num_experts):
            result[f"{prefix}.{i}.gate_proj.weight"] = gate_up[i, :half].contiguous()
            result[f"{prefix}.{i}.up_proj.weight"] = gate_up[i, half:].contiguous()
            result[f"{prefix}.{i}.down_proj.weight"] = down[i].contiguous()

        console.print(f"  Unfused [cyan]{num_experts}[/] experts at [dim]{prefix}[/]")

    orphaned = set(down_keys) - set(gate_up_keys)
    if orphaned:
        raise ValueError(
            f"Found down_proj without matching gate_up_proj at: {sorted(orphaned)}"
        )

    return result


def _resolve_verifier_path(verifier_path: Path) -> Path:
    """Return a local directory, downloading from HF if needed."""
    if verifier_path.exists():
        return verifier_path

    model_id = str(verifier_path)
    console.print(
        f"Verifier path [cyan]{model_id}[/] not found locally, "
        "downloading from HuggingFace..."
    )
    with _spinner() as progress:
        progress.add_task(f"Downloading {model_id}", total=None)
        local_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.index.json",
            ],
        )
    console.print(f"  Downloaded to [dim]{local_path}[/]")
    return Path(local_path)


def _load_finetuned_weights(
    checkpoint_dir: Path,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        shards = set(weight_map.values())
        with _bar() as progress:
            task = progress.add_task("Loading finetuned weights", total=len(shards))
            for shard in shards:
                with safe_open(str(checkpoint_dir / shard), framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        weights[key] = f.get_tensor(key)
                progress.advance(task)
        return weights

    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                weights[key] = f.get_tensor(key)
        return weights

    raise FileNotFoundError(f"No safetensors found at {checkpoint_dir}")


def _stitch_sharded(
    output_dir: Path,
    native_weights: dict[str, torch.Tensor],
) -> None:
    index_path = output_dir / "model.safetensors.index.json"
    with index_path.open() as f:
        index_data = json.load(f)
    weight_map: dict[str, str] = index_data["weight_map"]

    shard_to_new: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in native_weights.items():
        shard = weight_map.get(key)
        if shard is None:
            raise ValueError(
                f"Finetuned key '{key}' not found in verifier weight "
                "map. The finetuned checkpoint may not match the "
                "verifier."
            )
        shard_to_new.setdefault(shard, {})[key] = tensor

    with _bar() as progress:
        task = progress.add_task("Stitching shards", total=len(shard_to_new))
        for shard_filename, new_weights in shard_to_new.items():
            shard_path = output_dir / shard_filename
            existing: dict[str, torch.Tensor] = {}
            metadata = None
            with safe_open(str(shard_path), framework="pt") as f:
                metadata = f.metadata()
                for k in f.keys():  # noqa: SIM118
                    existing[k] = f.get_tensor(k)
            existing.update(new_weights)
            save_file(existing, str(shard_path), metadata=metadata)
            progress.advance(task)


def _stitch_single(
    safetensors_path: Path,
    native_weights: dict[str, torch.Tensor],
) -> None:
    existing: dict[str, torch.Tensor] = {}
    metadata = None
    with safe_open(str(safetensors_path), framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():  # noqa: SIM118
            existing[k] = f.get_tensor(k)
    existing.update(native_weights)
    save_file(existing, str(safetensors_path), metadata=metadata)


def stitch(
    finetuned_checkpoint: Path,
    verifier_path: Path,
    output_path: Path,
) -> Path:
    """Stitch finetuned MTP weights back into a verifier checkpoint."""
    console.print(
        Panel(
            f"[bold]Finetuned:[/] {finetuned_checkpoint}\n"
            f"[bold]Verifier:[/]  {verifier_path}\n"
            f"[bold]Output:[/]    {output_path}",
            title="[bold green]MTP Stitch[/]",
            border_style="green",
        )
    )

    verifier_path = _resolve_verifier_path(verifier_path)

    weights = _load_finetuned_weights(finetuned_checkpoint)
    console.print(f"  Loaded [cyan]{len(weights)}[/] finetuned weight tensors")

    frozen_count = sum(1 for k in weights if k in _FROZEN_KEYS)
    weights = _filter_frozen_keys(weights)
    if frozen_count:
        console.print(
            f"  Skipped [yellow]{frozen_count}[/] frozen key(s) (embed_tokens, lm_head)"
        )

    weights = _unfuse_moe_experts(weights)
    native_weights = {_remap_key(k): v for k, v in weights.items()}
    console.print(f"  Remapped [cyan]{len(native_weights)}[/] keys to native format")

    with _spinner() as progress:
        progress.add_task("Copying verifier checkpoint", total=None)
        shutil.copytree(verifier_path, output_path, dirs_exist_ok=True)

    index_path = output_path / "model.safetensors.index.json"
    if index_path.exists():
        _stitch_sharded(output_path, native_weights)
    else:
        single = output_path / "model.safetensors"
        if single.exists():
            _stitch_single(single, native_weights)
        else:
            raise FileNotFoundError(f"No safetensors checkpoint found at {output_path}")

    console.print(
        Panel(
            f"[bold]{output_path}[/]",
            title="[bold green]Stitched checkpoint saved[/]",
            border_style="green",
        )
    )
    return output_path


@app.command()
def main(
    finetuned_checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Path to the finetuned MTP speculator checkpoint.",
        ),
    ],
    verifier_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Path to the verifier checkpoint, or a HuggingFace "
                "model ID (e.g. Qwen/Qwen3-Next-80B-A3B-Instruct)."
            ),
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Output directory for the stitched checkpoint. "
                "Defaults to {verifier-name}-stitched."
            ),
        ),
    ] = None,
) -> None:
    """Stitch finetuned MTP weights back into a verifier checkpoint."""
    if output_path is None:
        output_path = Path.cwd() / f"{verifier_path.name}-stitched"

    stitch(
        finetuned_checkpoint=finetuned_checkpoint,
        verifier_path=verifier_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()
