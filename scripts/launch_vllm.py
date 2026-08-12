import argparse
import datetime
import hashlib
import json
import os
import shlex
import sys
import warnings
from pathlib import Path

from speculators.provenance import (
    atomic_write,
    find_package_repo,
    git_diff,
    pkg_version,
)
from speculators.provenance import (
    git_sha as _git_sha,
)

try:
    from hs_connectors import HiddenStatesBackend

    _backend_registry: dict[str, type[HiddenStatesBackend]] = dict(
        HiddenStatesBackend.registry  # type: ignore[misc]
    )
except ImportError:
    _backend_registry = {}  # type: ignore[assignment]


if "file" not in _backend_registry:
    # Vendored File backend in case hs_connectors is not available
    class _InlineFileBackend:
        @staticmethod
        def add_launch_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument(
                "--hidden-states-path",
                type=str,
                default="/tmp/hidden_states",  # noqa: S108
                help=(
                    "The directory to save hidden states to. "
                    "Default '/tmp/hidden_states'"
                ),
            )

        @staticmethod
        def build_kv_transfer_config(args: argparse.Namespace) -> dict:
            return {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": args.hidden_states_path,
                },
            }

    _backend_registry["file"] = _InlineFileBackend  # type: ignore[assignment]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for hidden states extraction",
        usage=(
            "launch_vllm.py [-h] MODEL [--hidden-states-backend BACKEND] "
            "[--target-layer-ids TARGET_LAYER_IDS [TARGET_LAYER_IDS ...]] -- *VLLM_ARGS"
        ),
    )
    parser.add_argument(
        "model", type=str, help="Model name or path to extract hidden states from"
    )

    parser.add_argument(
        "--hidden-states-backend",
        choices=list(_backend_registry.keys()),
        default="file",
        help=(
            "Hidden states transfer backend. Each backend may add its own "
            "CLI arguments (see below). Default: 'file'."
        ),
    )
    for backend_cls in _backend_registry.values():
        backend_cls.add_launch_args(parser)

    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of integer layer ids. Defaults to "
            "[2, num_hidden_layers // 2, num_hidden_layers - 3]. "
            "Note: if set, you must also pass the same value into the training process"
        ),
    )
    parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last layer (num_hidden_layers) to "
            "target_layer_ids for verifier hidden states extraction. Default: True"
        ),
    )
    parser.add_argument(
        "--provenance-dir",
        type=str,
        default=None,
        help=(
            "Directory to write vllm_command.txt, vllm.patch, and "
            "checkpoint_sha256.txt. Defaults to vllm_<model>_<timestamp>/."
        ),
    )
    parser.add_argument(
        "--no-hash-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Skip SHA256 hashing of .safetensors files. Useful for large "
            "checkpoints where hashing adds significant launch latency. "
            "When set, checkpoint_sha256.txt records file sizes and "
            "modification times instead."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    return parser.parse_known_args()


def _warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def _find_vllm_repo() -> str | None:
    """Find the vllm git checkout by walking up from the installed package."""
    repo = find_package_repo("vllm")
    if repo and (repo / "vllm" / "__init__.py").is_file():
        return str(repo)
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Individual provenance writers — each is self-contained and best-effort.
# ---------------------------------------------------------------------------


def _save_vllm_command(
    prov_dir: Path,
    cmd: list[str],
    sha: str,
    diff: str,
    vllm_ver: str,
) -> None:
    sha_label = f"{sha} (dirty)" if diff else sha
    ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    header = "\n".join(
        [
            f"# Timestamp: {ts}",
            f"# Python: {sys.executable}",
            f"# Git SHA: {sha_label}",
            f"# vllm: {vllm_ver}",
        ]
    )
    atomic_write(
        prov_dir / "vllm_command.txt",
        f"{header}\n{shlex.join(cmd)}\n",
    )


def _save_vllm_patch(
    prov_dir: Path,
    vllm_repo: str | None,
    sha: str,
    diff: str,
    vllm_ver: str,
) -> None:
    if vllm_repo:
        content = f"# repo: {vllm_repo} ({sha})\n{diff}"
    else:
        content = f"# vllm {vllm_ver} (wheel install, no git repo found)\n"
    atomic_write(prov_dir / "vllm.patch", content)


def _save_checkpoint_sha256(
    prov_dir: Path, model: str, *, skip_hash: bool = False
) -> None:
    dest = prov_dir / "checkpoint_sha256.txt"
    model_path = os.path.expanduser(model)
    if not os.path.isdir(model_path):
        atomic_write(dest, f"# model: {model} (not a local path)\n")
        return
    safetensors = sorted(
        f for f in os.listdir(model_path) if f.endswith(".safetensors")
    )
    if not safetensors:
        atomic_write(dest, f"# no .safetensors files in {model_path}\n")
        return
    if skip_hash:
        lines = []
        for name in safetensors:
            fp = os.path.join(model_path, name)
            st = os.stat(fp)
            lines.append(f"size={st.st_size}  mtime={st.st_mtime}  {name}")
        header = "# hashing skipped (--no-hash-checkpoints)\n"
        atomic_write(dest, header + "\n".join(lines) + "\n")
    else:
        lines = [
            f"{_sha256_file(os.path.join(model_path, name))}  {name}"
            for name in safetensors
        ]
        atomic_write(dest, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _save_vllm_provenance(
    cmd: list[str],
    provenance_dir: str,
    model: str,
    *,
    skip_hash: bool = False,
) -> None:
    """Write vllm_command.txt, vllm.patch, and checkpoint_sha256.txt.

    Best-effort — failures warn but never block the vLLM launch.
    """
    prov_dir = Path(provenance_dir)
    try:
        prov_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _warn(f"could not create provenance dir: {exc}")
        return

    vllm_repo = _find_vllm_repo()
    vllm_root = Path(vllm_repo) if vllm_repo else None
    sha = _git_sha(vllm_root)
    diff = git_diff(vllm_root)
    vllm_ver = pkg_version("vllm")

    writers = [
        (
            "vllm_command.txt",
            lambda: _save_vllm_command(prov_dir, cmd, sha, diff, vllm_ver),
        ),
        (
            "vllm.patch",
            lambda: _save_vllm_patch(prov_dir, vllm_repo, sha, diff, vllm_ver),
        ),
        (
            "checkpoint_sha256.txt",
            lambda: _save_checkpoint_sha256(prov_dir, model, skip_hash=skip_hash),
        ),
    ]
    for artifact, write in writers:
        try:
            write()
        except Exception as exc:  # noqa: BLE001
            _warn(f"could not save {artifact}: {exc}")


def main():
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")

    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(args.model)
    if hasattr(config, "text_config"):
        config = config.text_config
    num_hidden_layers = config.num_hidden_layers

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly passed into the training script.",
            stacklevel=2,
        )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    backend_cls = _backend_registry[args.hidden_states_backend]
    kv_transfer_config = backend_cls.build_kv_transfer_config(args)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config",
        json.dumps(speculative_config),
        "--kv_transfer_config",
        json.dumps(kv_transfer_config),
        *vllm_args,
    ]

    print("Running command:")
    print(" ".join(cmd))

    if not args.provenance_dir:
        sanitized = args.model.replace("/", "_").replace(" ", "_")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.provenance_dir = f"vllm_{sanitized}_{ts}"
    _save_vllm_provenance(
        cmd,
        args.provenance_dir,
        args.model,
        skip_hash=args.no_hash_checkpoints,
    )

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
