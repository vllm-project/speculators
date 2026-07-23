#!/usr/bin/env python3
"""
Export a trained Gemma4 MTP speculator as a standalone vLLM-compatible checkpoint.

Unlike stitch_mtp.py (which merges MTP weights INTO a verifier checkpoint for
Qwen-style models), this script produces a SEPARATE checkpoint directory that
vLLM loads as an independent draft model via the Gemma4Speculator path.

Usage:
    python scripts/export_gemma4_mtp.py ./finetuned-mtp google/gemma-4-31B-it

    # custom output path (defaults to {finetuned}-exported):
    python scripts/export_gemma4_mtp.py ./finetuned-mtp ./verifier --output-path ./out
"""

import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from safetensors import safe_open
from safetensors.torch import save_file

app = typer.Typer(rich_markup_mode="rich")
console = Console()

_FROZEN_KEYS = {"embed_tokens.weight", "lm_head.weight"}

_EXACT_REMAP: dict[str, str] = {
    "mtp_layers.0.pre_projection.": "pre_projection.",
    "mtp_layers.0.post_projection.": "post_projection.",
    "mtp_layers.0.final_norm.": "model.norm.",
}

_PREFIX_REMAP = "mtp_layers.0."
_PREFIX_TARGET = "model.layers.0."

_TEXT_CONFIG_FIELDS = [
    "attention_bias",
    "attention_dropout",
    "attention_k_eq_v",
    "bos_token_id",
    "eos_token_id",
    "final_logit_softcapping",
    "global_head_dim",
    "head_dim",
    "hidden_activation",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_key_value_heads",
    "num_global_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_parameters",
    "sliding_window",
    "vocab_size",
]


def _spinner() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )


def _resolve_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_id = str(model_path)
    console.print(
        f"Path [cyan]{model_id}[/] not found locally, downloading from HuggingFace..."
    )
    with _spinner() as progress:
        progress.add_task(f"Downloading {model_id}", total=None)
        local_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
        )
    console.print(f"  Downloaded to [dim]{local_path}[/]")
    return Path(local_path)


def _load_safetensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        for shard in set(weight_map.values()):
            with safe_open(str(checkpoint_dir / shard), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    weights[key] = f.get_tensor(key)
        return weights

    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                weights[key] = f.get_tensor(key)
        return weights

    raise FileNotFoundError(f"No safetensors found at {checkpoint_dir}")


def _remap_key(key: str) -> str:
    for src, dst in _EXACT_REMAP.items():
        if key.startswith(src):
            return dst + key[len(src):]

    if key.startswith(_PREFIX_REMAP):
        return _PREFIX_TARGET + key[len(_PREFIX_REMAP):]

    if key == "embed_tokens.weight":
        return "model.embed_tokens.weight"

    return key


def _load_verifier_shared_weights(
    verifier_dir: Path,
) -> dict[str, torch.Tensor]:
    """Load only embed_tokens and lm_head from the verifier."""
    shared: dict[str, torch.Tensor] = {}
    target_keys = {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.lm_head.weight",
        "language_model.lm_head.weight",
    }

    index_path = verifier_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        relevant_shards = {
            weight_map[k] for k in target_keys if k in weight_map
        }
        for shard in relevant_shards:
            with safe_open(str(verifier_dir / shard), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    if key in target_keys:
                        shared[key] = f.get_tensor(key)
    else:
        single = verifier_dir / "model.safetensors"
        if single.exists():
            with safe_open(str(single), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    if key in target_keys:
                        shared[key] = f.get_tensor(key)

    result: dict[str, torch.Tensor] = {}
    for key, tensor in shared.items():
        if "embed_tokens" in key:
            result["model.embed_tokens.weight"] = tensor
        elif "lm_head" in key:
            result["lm_head.weight"] = tensor

    return result


def _build_config(verifier_dir: Path) -> dict:
    with (verifier_dir / "config.json").open() as f:
        verifier_config = json.load(f)

    verifier_text = verifier_config.get("text_config", verifier_config)

    text_config: dict = {}
    for field in _TEXT_CONFIG_FIELDS:
        if field in verifier_text:
            text_config[field] = verifier_text[field]

    text_config["num_hidden_layers"] = 1
    text_config["layer_types"] = ["full_attention"]
    text_config["enable_moe_block"] = False
    text_config["hidden_size_per_layer_input"] = 0
    text_config["vocab_size_per_layer_input"] = 0
    text_config["tie_word_embeddings"] = True

    hidden_size = verifier_text["hidden_size"]
    config = {
        "model_type": "gemma4_assistant",
        "architectures": ["Gemma4MTPModel"],
        "backbone_hidden_size": hidden_size,
        "tie_word_embeddings": True,
        "text_config": text_config,
    }
    return config


def export(
    finetuned_checkpoint: Path,
    verifier_path: Path,
    output_path: Path,
) -> Path:
    console.print(
        Panel(
            f"[bold]Finetuned:[/] {finetuned_checkpoint}\n"
            f"[bold]Verifier:[/]  {verifier_path}\n"
            f"[bold]Output:[/]    {output_path}",
            title="[bold green]Gemma4 MTP Export[/]",
            border_style="green",
        )
    )

    verifier_path = _resolve_path(verifier_path)

    with _spinner() as progress:
        progress.add_task("Loading finetuned weights", total=None)
        finetuned = _load_safetensors(finetuned_checkpoint)
    console.print(f"  Loaded [cyan]{len(finetuned)}[/] finetuned weight tensors")

    trained_keys = {k: v for k, v in finetuned.items() if k not in _FROZEN_KEYS}
    console.print(
        f"  Filtered to [cyan]{len(trained_keys)}[/] trained keys "
        f"(skipped {len(finetuned) - len(trained_keys)} frozen)"
    )

    exported: dict[str, torch.Tensor] = {}
    for key, tensor in trained_keys.items():
        new_key = _remap_key(key)
        exported[new_key] = tensor
    console.print(f"  Remapped [cyan]{len(exported)}[/] keys to vLLM format")

    with _spinner() as progress:
        progress.add_task("Loading verifier shared weights", total=None)
        shared = _load_verifier_shared_weights(verifier_path)
    exported.update(shared)
    console.print(f"  Added [cyan]{len(shared)}[/] shared weight(s) from verifier")

    output_path.mkdir(parents=True, exist_ok=True)

    config = _build_config(verifier_path)
    config_path = output_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    console.print(f"  Wrote config to [dim]{config_path}[/]")

    weights_path = output_path / "model.safetensors"
    with _spinner() as progress:
        progress.add_task("Saving exported weights", total=None)
        save_file(exported, str(weights_path))
    console.print(f"  Saved [cyan]{len(exported)}[/] tensors to [dim]{weights_path}[/]")

    console.print("\n[bold]Exported weight keys:[/]")
    for key in sorted(exported.keys()):
        shape = list(exported[key].shape)
        console.print(f"  [dim]{key}[/]: {shape}")

    console.print(
        Panel(
            f"[bold]{output_path}[/]\n\n"
            "Deploy with:\n"
            f"  vllm serve <verifier> --speculative-config "
            f"'{{\"method\":\"mtp\",\"model\":\"{output_path}\","
            f"\"num_speculative_tokens\":3}}'",
            title="[bold green]Export complete[/]",
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
                "model ID (e.g. google/gemma-4-31B-it)."
            ),
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Output directory for the exported checkpoint. "
                "Defaults to {finetuned}-exported."
            ),
        ),
    ] = None,
) -> None:
    """Export a trained Gemma4 MTP speculator for vLLM deployment."""
    if output_path is None:
        output_path = Path.cwd() / f"{finetuned_checkpoint.name}-exported"

    export(
        finetuned_checkpoint=finetuned_checkpoint,
        verifier_path=verifier_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()
