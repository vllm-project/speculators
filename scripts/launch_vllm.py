import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
import warnings

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
            "Directory to write vllm_command.txt and vllm.patch for "
            "eval reproducibility (e.g. the eval output directory)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    return parser.parse_known_args()


def _atomic_write(path: str, content: str) -> None:
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path),
        prefix=f".{os.path.basename(path)}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _git_info(repo: str) -> tuple[str, str | None]:
    """Return ``(sha, diff)`` for a git repo.  *diff* is None on failure."""
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=5,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        sha = "unknown"

    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=repo,
            timeout=30,
            check=False,
        )
        diff = result.stdout if result.returncode == 0 else None
    except OSError:
        diff = None

    return sha, diff


def _is_vllm_repo(path: str) -> bool:
    """Check that *path* looks like the vllm source tree, not an unrelated repo."""
    return os.path.isdir(os.path.join(path, ".git")) and os.path.isfile(
        os.path.join(path, "vllm", "__init__.py")
    )


def _find_vllm_repo() -> str | None:
    """Find the vllm git checkout by walking up from the installed package."""
    try:
        dist = importlib.metadata.distribution("vllm")
        if dist.files:
            d = str(dist._path.parent)
            while d != os.path.dirname(d):
                if _is_vllm_repo(d):
                    return d
                d = os.path.dirname(d)
    except (importlib.metadata.PackageNotFoundError, AttributeError):
        pass

    for candidate in [
        os.path.expanduser("~/vllm"),
        "/workspace/vllm",
    ]:
        if _is_vllm_repo(candidate):
            return candidate
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _save_checkpoint_sha256(provenance_dir: str, model: str) -> None:
    dest = os.path.join(provenance_dir, "checkpoint_sha256.txt")
    try:
        model_path = os.path.expanduser(model)
        if os.path.isdir(model_path):
            safetensors = sorted(
                f for f in os.listdir(model_path) if f.endswith(".safetensors")
            )
            if safetensors:
                lines = []
                for name in safetensors:
                    h = _sha256_file(os.path.join(model_path, name))
                    lines.append(f"{h}  {name}")
                _atomic_write(dest, "\n".join(lines) + "\n")
            else:
                _atomic_write(
                    dest,
                    f"# no .safetensors files in {model_path}\n",
                )
        else:
            _atomic_write(dest, f"# model: {model} (not a local path)\n")
    except OSError as exc:
        print(
            f"Warning: could not save checkpoint_sha256.txt: {exc}",
            file=sys.stderr,
        )


def _save_vllm_provenance(cmd: list[str], provenance_dir: str, model: str) -> None:
    """Write vllm_command.txt and vllm.patch to *provenance_dir*.

    Best-effort — failures are printed but never fatal.
    """
    try:
        os.makedirs(provenance_dir, exist_ok=True)
    except OSError as exc:
        print(
            f"Warning: could not create provenance dir: {exc}",
            file=sys.stderr,
        )
        return

    vllm_repo = _find_vllm_repo()
    git_sha = "unknown"
    diff: str | None = None
    if vllm_repo:
        git_sha, diff = _git_info(vllm_repo)

    try:
        vllm_ver = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_ver = "unknown"

    # --- vllm_command.txt ---
    try:
        sha_label = git_sha
        if diff:
            sha_label += " (dirty)"

        ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        header = "\n".join(
            [
                f"# Timestamp: {ts}",
                f"# Python: {sys.executable}",
                f"# Git SHA: {sha_label}",
                f"# vllm: {vllm_ver}",
            ]
        )
        content = f"{header}\n{shlex.join(cmd)}\n"
        _atomic_write(os.path.join(provenance_dir, "vllm_command.txt"), content)
    except OSError as exc:
        print(
            f"Warning: could not save vllm_command.txt: {exc}",
            file=sys.stderr,
        )

    # --- vllm.patch ---
    try:
        if vllm_repo:
            patch_content = f"# repo: {vllm_repo} ({git_sha})\n"
            if diff:
                patch_content += diff
            _atomic_write(
                os.path.join(provenance_dir, "vllm.patch"),
                patch_content,
            )
        else:
            _atomic_write(
                os.path.join(provenance_dir, "vllm.patch"),
                f"# vllm {vllm_ver} (wheel install, no git repo found)\n",
            )
    except OSError as exc:
        print(
            f"Warning: could not save vllm.patch: {exc}",
            file=sys.stderr,
        )

    _save_checkpoint_sha256(provenance_dir, model)


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

    disable_cp_arg = "--no-enable-chunked-prefill"
    if disable_cp_arg not in cmd:
        cmd.append(disable_cp_arg)

    print("Running command:")
    print(" ".join(cmd))

    if args.provenance_dir:
        _save_vllm_provenance(cmd, args.provenance_dir, args.model)

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
