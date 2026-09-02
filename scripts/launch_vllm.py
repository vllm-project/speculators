import argparse
import datetime
import hashlib
import json
import os
import shlex
import sys
import warnings
from pathlib import Path

from _provenance import (
    atomic_write,
    find_package_repo,
    git_diff,
    pkg_version,
)
from _provenance import (
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


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provenance-dir",
        type=str,
        default=None,
        help=(
            "Directory to write vllm_command.txt, vllm.patch, and "
            "checkpoint_sha256.txt (plus drafter_checkpoint_sha256.txt in "
            "eval mode). Provenance is only captured when this is set; omit "
            "it to skip logging (e.g. for ad-hoc test/debug runs)."
        ),
    )
    parser.add_argument(
        "--no-hash-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Skip SHA256 hashing of .safetensors files. Useful "
            "for large checkpoints where hashing adds significant "
            "launch latency. When set, checkpoint_sha256.txt "
            "records file sizes and modification times instead."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for training or evaluation",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # --- train subcommand (default when no subcommand given) ---
    train_parser = sub.add_parser(
        "train",
        help="Hidden-states extraction for training data generation",
    )
    train_parser.add_argument(
        "model",
        type=str,
        help="Model name or path to extract hidden states from",
    )
    train_parser.add_argument(
        "--hidden-states-backend",
        choices=list(_backend_registry.keys()),
        default="file",
        help=(
            "Hidden states transfer backend. Each backend may "
            "add its own CLI arguments (see below). "
            "Default: 'file'."
        ),
    )
    for backend_cls in _backend_registry.values():
        backend_cls.add_launch_args(train_parser)
    train_parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) Space-separated list of integer layer "
            "ids. Defaults to "
            "[2, num_hidden_layers // 2, num_hidden_layers - 3]."
            " Note: if set, you must also pass the same value "
            "into the training process"
        ),
    )
    train_parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last layer (num_hidden_layers) to "
            "target_layer_ids for verifier hidden states "
            "extraction. Default: True"
        ),
    )
    train_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow custom model configuration code while resolving "
            "hidden-state layer ids. Pass the same flag after '--' for "
            "vLLM itself."
        ),
    )
    _add_shared_args(train_parser)

    # --- eval subcommand ---
    eval_parser = sub.add_parser(
        "eval",
        help="Speculative decoding serving for evaluation",
    )
    eval_parser.add_argument(
        "model",
        type=str,
        help="Target model name or path",
    )
    eval_parser.add_argument(
        "--spec-model",
        type=str,
        required=True,
        help="Drafter model name or path",
    )
    eval_parser.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="Number of speculative tokens",
    )
    eval_parser.add_argument(
        "--spec-method",
        type=str,
        default=None,
        help="Speculative decoding method (optional)",
    )
    _add_shared_args(eval_parser)

    subcommands = set(sub.choices)
    argv = sys.argv[1:]
    if not argv or (argv[0] not in subcommands and not argv[0].startswith("-")):
        argv = ["train", *argv]
    args, vllm_args = parser.parse_known_args(argv)
    if args.subcommand is None:
        args, vllm_args = parser.parse_known_args(["train"] + sys.argv[1:])
    return args, vllm_args


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
        content = f"# repo: {vllm_repo} ({sha})\n{diff}\n"
    else:
        content = f"# vllm {vllm_ver} (wheel install, no git repo found)\n"
    atomic_write(prov_dir / "vllm.patch", content)


def _save_checkpoint_sha256(
    prov_dir: Path,
    model: str,
    *,
    skip_hash: bool = False,
    dest_name: str = "checkpoint_sha256.txt",
) -> None:
    dest = prov_dir / dest_name
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
    spec_model: str | None = None,
) -> None:
    """Write vllm_command.txt, vllm.patch, and checkpoint_sha256.txt.

    In eval mode (``spec_model`` given) also writes
    drafter_checkpoint_sha256.txt. Best-effort — failures warn but never
    block the vLLM launch.
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
    if spec_model is not None:
        drafter_dest = "drafter_checkpoint_sha256.txt"
        writers.append(
            (
                drafter_dest,
                lambda: _save_checkpoint_sha256(
                    prov_dir,
                    spec_model,
                    skip_hash=skip_hash,
                    dest_name=drafter_dest,
                ),
            )
        )
    for artifact, write in writers:
        try:
            write()
        except Exception as exc:  # noqa: BLE001
            _warn(f"could not save {artifact}: {exc}")


def _build_train_cmd(args, vllm_args):
    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
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
    # Layer id ``num_hidden_layers`` (the final hidden state) is valid: the
    # default above and --include-last-layer both emit it.
    if (
        min(target_layer_ids) < 0
        or max(target_layer_ids) > num_hidden_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    ):
        raise ValueError(
            f"Invalid target layer ids {target_layer_ids}; ids must be "
            f"distinct and within [0, {num_hidden_layers}]."
        )

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    backend_cls = _backend_registry[args.hidden_states_backend]
    kv_transfer_config = backend_cls.build_kv_transfer_config(args)

    return [
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


def _build_eval_cmd(args, vllm_args):
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--spec-model",
        args.spec_model,
    ]
    if args.spec_tokens is not None:
        cmd.extend(["--spec-tokens", str(args.spec_tokens)])
    if args.spec_method is not None:
        cmd.extend(["--spec-method", args.spec_method])
    cmd.extend(vllm_args)
    return cmd


def main():
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")

    if args.subcommand == "train":
        cmd = _build_train_cmd(args, vllm_args)
    elif args.subcommand == "eval":
        cmd = _build_eval_cmd(args, vllm_args)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    print("Running command:")
    print(" ".join(cmd))

    if args.provenance_dir:
        _save_vllm_provenance(
            cmd,
            args.provenance_dir,
            args.model,
            skip_hash=args.no_hash_checkpoints,
            spec_model=getattr(args, "spec_model", None),
        )

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
