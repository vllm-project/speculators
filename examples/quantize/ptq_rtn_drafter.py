"""Calibration-free round-to-nearest (RTN) quantization of a speculators drafter.

Quantizes the projection weights of a trained speculator checkpoint with
compressed-tensors and saves the result in compressed-tensors format, which vLLM
loads directly as the draft model (the drafter's ``config.json`` carries its own
``quantization_config``).

The drafter is not a ``transformers.PreTrainedModel``, so instead of loading the
speculators model class this script rebuilds a plain ``nn.Module`` skeleton from
the checkpoint's tensor names. Parameter names therefore match the original
checkpoint one to one, which is what vLLM's speculator loaders expect.

What gets quantized:

- every 2-D ``*_proj.weight`` and ``fc.weight`` (all decoder-layer attention and
  MLP projections plus the input fusion layer)
- kept in the original dtype: norms, ``embed_tokens``, ``lm_head``, and the DSpark
  ``markov_head`` / ``confidence_head``. The ignore list uses regexes so it also
  matches vLLM's ``model.``-prefixed module names.

Scales are plain min/max (no calibration data, no GPTQ/AWQ). Any compressed-tensors
preset with weight-only or dynamic activation quantization works; tested presets:

- ``FP8_DYNAMIC``: FP8 E4M3 per-channel weights, dynamic per-token FP8 activations
- ``W4A16``: INT4 symmetric weights, group size 128
- ``NVFP4A16``: FP4 E2M1 weights, FP8 group-16 scales x FP32 global scale
- ``MXFP4A16``: FP4 E2M1 weights, E8M0 group-32 scales

For NVFP4, vLLM fuses ``q/k/v_proj`` and ``gate/up_proj`` into one layer that
keeps a single weight global scale, so the members of each fused group are given
a shared global scale here (the minimum, as llm-compressor does).

Requires ``compressed-tensors>=0.20.1``. Like the training and eval scripts, it
writes its command line and package versions to ``quantize_command.txt``.

Example:

    python examples/quantize/ptq_rtn_drafter.py \
        --model RedHatAI/Qwen3.8-27B-speculator.dspark \
        --scheme FP8_DYNAMIC \
        --output Qwen3.8-27B-speculator.dspark-FP8_DYNAMIC
"""

import argparse
import json
import math
import shlex
import shutil
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStrategy,
    apply_quantization_config,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn

from speculators.provenance import (
    atomic_write,
    find_package_repo,
    git_sha,
    package_versions,
)

IGNORE = [
    "re:.*markov_head.*",
    "re:.*confidence_head.*",
    "re:.*lm_head.*",
    "re:.*embed_tokens.*",
]
LINEAR_SUFFIXES = ("_proj.weight", "fc.weight")
# Files copied next to the quantized weights. Training artifacts of the source
# run (train_command.txt, training_state.json, ...) and its model card are not
# copied, since they describe the bf16 checkpoint.
COPY_PATTERNS = ("config.json", "generation_config.json", "*.py")
# Layers vLLM fuses into one linear layer with a single NVFP4 weight global scale.
FUSED_GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))


class _ParamHolder(nn.Module):
    """Holds a non-linear parameter (e.g. a norm weight) under its original name."""

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(tensor, requires_grad=False)


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    *parents, leaf = path.split(".")
    current = root
    for name in parents:
        if not hasattr(current, name):
            current.add_module(name, nn.Module())
        current = getattr(current, name)
    current.add_module(leaf, module)


def build_skeleton(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    """Rebuild a module tree whose ``state_dict`` keys equal the checkpoint's.

    2-D projection weights become ``nn.Linear`` (the quantization targets), other
    2-D tensors become ``nn.Embedding`` (never targeted), everything else is held
    as a plain parameter.
    """
    root = nn.Module()
    for name, tensor in state_dict.items():
        if not name.endswith(".weight"):
            continue  # biases are attached to their Linear below
        module_name = name.removesuffix(".weight")
        bias = state_dict.get(f"{module_name}.bias")
        if tensor.ndim == 2 and (name.endswith(LINEAR_SUFFIXES) or bias is not None):
            linear = nn.Linear(
                tensor.shape[1],
                tensor.shape[0],
                bias=bias is not None,
                dtype=tensor.dtype,
                device="meta",
            )
            linear.weight = nn.Parameter(tensor, requires_grad=False)
            if bias is not None:
                linear.bias = nn.Parameter(bias, requires_grad=False)
            _set_submodule(root, module_name, linear)
        elif tensor.ndim == 2:
            embedding = nn.Embedding(*tensor.shape, dtype=tensor.dtype, device="meta")
            embedding.weight = nn.Parameter(tensor, requires_grad=False)
            _set_submodule(root, module_name, embedding)
        else:
            _set_submodule(root, module_name, _ParamHolder(tensor))

    missing = set(state_dict) - set(root.state_dict())
    if missing:
        raise ValueError(f"Could not place tensors in the skeleton: {sorted(missing)}")
    return root


def _is_tensor_group(module: nn.Module) -> bool:
    scheme = getattr(module, "quantization_scheme", None)
    return (
        scheme is not None
        and scheme.weights is not None
        and scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP
    )


def fused_global_scales(model: nn.Module) -> dict[str, torch.Tensor]:
    """NVFP4 weight global scale per module; fused groups share their minimum.

    Taking the minimum keeps every member's FP8 group scales in range.
    """
    scales = {
        name: generate_gparam(*torch.aminmax(module.weight.data.float()))
        for name, module in model.named_modules()
        if _is_tensor_group(module)
    }
    shared = dict(scales)
    for name in scales:
        parent, _, leaf = name.rpartition(".")
        for group in FUSED_GROUPS:
            if leaf not in group:
                continue
            members = [f"{parent}.{g}" if parent else g for g in group]
            if all(member in scales for member in members):
                shared[name] = torch.stack([scales[m] for m in members]).amin(dim=0)
    return shared


def set_rtn_qparams(module: nn.Linear, global_scale: torch.Tensor | None) -> None:
    """Write min/max (round-to-nearest) weight scales and zero points."""
    args = module.quantization_scheme.weights
    weight = module.weight.data.float()

    if args.strategy == QuantizationStrategy.TENSOR_GROUP:
        if global_scale is None:
            raise ValueError("TENSOR_GROUP quantization needs a global scale")
        module.weight_global_scale.copy_(
            global_scale.reshape(module.weight_global_scale.shape)
        )

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        groups = weight.unflatten(
            -1, (math.ceil(weight.shape[-1] / args.group_size), args.group_size)
        )
        min_vals, max_vals = groups.amin(dim=-1), groups.amax(dim=-1)
    elif args.strategy == QuantizationStrategy.CHANNEL:
        min_vals, max_vals = weight.amin(dim=-1), weight.amax(dim=-1)
    elif args.strategy == QuantizationStrategy.TENSOR:
        min_vals, max_vals = torch.aminmax(weight)
    else:
        raise NotImplementedError(f"Unsupported weight strategy: {args.strategy}")

    scale, zero_point = calculate_qparams(
        min_vals, max_vals, args, global_scale=global_scale
    )
    module.weight_scale.copy_(scale.reshape(module.weight_scale.shape))
    if getattr(module, "weight_zero_point", None) is not None:
        module.weight_zero_point.copy_(
            zero_point.reshape(module.weight_zero_point.shape)
        )


def quantize_weights(model: nn.Module) -> tuple[int, dict[str, dict[str, float]]]:
    """Set RTN qparams on every targeted layer.

    :return: number of quantized parameters and, per layer type, the mean and
        max relative Frobenius error of the fake-quantized weights.
    """
    global_scales = fused_global_scales(model)  # before any scale is written
    errors: dict[str, list[float]] = defaultdict(list)
    num_quantized = 0
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        set_rtn_qparams(module, global_scales.get(name))
        weight = module.weight.data
        fake_quantized = fake_quantize(
            weight,
            module.weight_scale,
            getattr(module, "weight_zero_point", None),
            scheme.weights,
            global_scale=getattr(module, "weight_global_scale", None),
        )
        rel_err = (
            fake_quantized.float() - weight.float()
        ).norm() / weight.float().norm()
        errors[name.rpartition(".")[2]].append(rel_err.item())
        num_quantized += weight.numel()

    stats = {
        kind: {"count": len(v), "mean_rel_err": sum(v) / len(v), "max_rel_err": max(v)}
        for kind, v in errors.items()
    }
    return num_quantized, stats


def resolve_model_dir(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    return Path(
        snapshot_download(model, allow_patterns=["*.json", "*.py", "*.safetensors"])
    )


def load_state_dict(model_dir: Path, device: str) -> dict[str, torch.Tensor]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No *.safetensors files in {model_dir}")
    state_dict: dict[str, torch.Tensor] = {}
    for file in files:
        state_dict.update(load_file(str(file), device=device))
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model", required=True, help="Speculator checkpoint: local dir or HF repo id"
    )
    parser.add_argument(
        "--scheme",
        required=True,
        help="compressed-tensors preset, e.g. FP8_DYNAMIC, W4A16, NVFP4A16, MXFP4A16",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    start = time.time()
    src, out = resolve_model_dir(args.model), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    scheme = preset_name_to_scheme(args.scheme.upper(), targets=["Linear"])
    activations = scheme.input_activations
    if activations is not None and activations.dynamic is not True:
        # static (and NVFP4 "local") activation scales need calibration data
        raise ValueError(
            f"{args.scheme} needs calibrated activation scales; this script only "
            "supports weight-only or fully dynamic activation quantization"
        )

    model = build_skeleton(load_state_dict(src, args.device))
    num_params = sum(p.numel() for p in model.parameters())

    apply_quantization_config(
        model, QuantizationConfig(config_groups={"group_0": scheme}, ignore=IGNORE)
    )
    num_quantized, stats = quantize_weights(model)
    print(
        f"Quantized {num_quantized / 1e9:.3f}B of {num_params / 1e9:.3f}B "
        f"parameters with {args.scheme}"
    )
    for kind, s in stats.items():
        print(
            f"  {kind:10s} n={s['count']:3d} rel_err "
            f"mean={s['mean_rel_err']:.4f} max={s['max_rel_err']:.4f}"
        )

    compressor = ModelCompressor.from_pretrained_model(model)
    qconfig = compressor.quantization_config
    qconfig.ignore = sorted(set(qconfig.ignore or []) | set(IGNORE))
    compressor.compress_model(model)

    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for pattern in COPY_PATTERNS:
        for file in src.glob(pattern):
            shutil.copy(file, out / file.name)
    compressor.update_config(str(out))  # adds quantization_config to config.json

    atomic_write(
        out / "quantize_command.txt",
        "\n".join(
            [
                f"# Timestamp: {datetime.now(UTC).isoformat()}",
                f"# Git SHA: {git_sha(find_package_repo('speculators'))}",
                *package_versions(),
                shlex.join(sys.argv),
            ]
        )
        + "\n",
    )
    with (out / "ptq_rtn_stats.json").open("w") as f:
        json.dump(
            {
                "scheme": args.scheme,
                "source": args.model,
                "num_quantized_params": num_quantized,
                "num_params": num_params,
                "format": str(qconfig.format),
                "ignore": qconfig.ignore,
                "rel_err": stats,
            },
            f,
            indent=2,
        )
    size_gb = (out / "model.safetensors").stat().st_size / 1e9
    print(
        f"Saved {out} ({size_gb:.2f} GB, format={qconfig.format}) "
        f"in {time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
