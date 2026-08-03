import argparse
import json
import os
import sys
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
        help="Directory to save provenance artifacts (vllm_command.txt, vllm.patch, checkpoint_sha256.txt).",
    )
    parser.add_argument(
        "--no-hash-checkpoints",
        action="store_true",
        help="Skip hashing the .safetensors checkpoints in the provenance directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    return parser.parse_known_args()


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

    import datetime
    import shlex
    from pathlib import Path
    from speculators.utils.provenance import (
        get_git_sha,
        get_git_diff,
        get_package_root,
        get_package_versions,
        hash_safetensors,
        write_atomic,
    )

    prov_dir = args.provenance_dir
    if not prov_dir:
        safe_model = args.model.replace("/", "_")
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        prov_dir = f"vllm_{safe_model}_{ts}"
    prov_path = Path(prov_dir)

    spec_root = get_package_root("speculators") or Path.cwd()
    vllm_root = get_package_root("vllm")
    
    spec_sha = get_git_sha(spec_root)
    versions = get_package_versions(["vllm"])
    vllm_ver = versions[0] if versions else "unknown"

    vllm_cmd_header = "\n".join([
        f"# Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"# Python: {sys.executable}",
        f"# Speculators Git SHA: {spec_sha}",
        f"{vllm_ver}",
    ])
    vllm_cmd_content = f"{vllm_cmd_header}\n{shlex.join(cmd)}\n"
    write_atomic(prov_path, "vllm_command.txt", vllm_cmd_content)

    if vllm_root and (vllm_root.parent / ".git").exists():
        vllm_repo = vllm_root.parent
        diff = get_git_diff(vllm_repo)
        patch_header = f"# repo: {vllm_repo} ({get_git_sha(vllm_repo)})"
        write_atomic(prov_path, "vllm.patch", f"{patch_header}\n{diff}\n")
    else:
        write_atomic(prov_path, "vllm.patch", f"# Installed from wheel or non-git source\n{vllm_ver}\n")

    if not args.no_hash_checkpoints:
        hashes = hash_safetensors(args.model)
        write_atomic(prov_path, "checkpoint_sha256.txt", hashes + "\n")

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
