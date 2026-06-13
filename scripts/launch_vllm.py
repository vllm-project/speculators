import argparse
import json
import os
import socket
import sys
import warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for hidden states extraction",
        usage=(
            "launch_vllm.py [-h] MODEL [--hidden-states-path HIDDEN_STATES_PATH] "
            "[--target-layer-ids TARGET_LAYER_IDS [TARGET_LAYER_IDS ...]] -- *VLLM_ARGS"
        ),
    )
    parser.add_argument(
        "model", type=str, help="Model name or path to extract hidden states from"
    )
    parser.add_argument(
        "--hidden-states-path",
        type=str,
        default="/tmp/hidden_states",  # noqa: S108
        help="The directory to save hidden states to. Default '/tmp/hidden_states'.",
    )
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
        "--hidden-states-backend",
        choices=["file", "mooncake"],
        default="file",
        help=(
            "Transport for extracted hidden states. 'file' writes safetensors to "
            "--hidden-states-path (requires a shared filesystem). 'mooncake' writes "
            "to a Mooncake distributed store, enabling the target and trainer to run "
            "on different nodes with no shared filesystem. Default: 'file'."
        ),
    )
    parser.add_argument(
        "--mooncake-master",
        type=str,
        default="localhost:50051",
        help="Mooncake master server address (host:port). Used with backend=mooncake.",
    )
    parser.add_argument(
        "--mooncake-metadata-server",
        type=str,
        default="http://localhost:8080/metadata",
        help="Mooncake metadata server URL. Used with backend=mooncake.",
    )
    parser.add_argument(
        "--mooncake-protocol",
        choices=["tcp", "rdma"],
        default="tcp",
        help="Mooncake transport protocol. Used with backend=mooncake.",
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
    if args.hidden_states_backend == "mooncake":
        # Out-of-tree connector: vLLM imports it via kv_connector_module_path,
        # so no registration in vLLM's factory is needed.
        kv_transfer_config = {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_connector_module_path": (
                "speculators.data_generation.mooncake_hidden_states_connector"
            ),
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "mooncake": {
                    "local_hostname": socket.gethostbyname(socket.gethostname()),
                    "master_server_address": args.mooncake_master,
                    "metadata_server": args.mooncake_metadata_server,
                    "protocol": args.mooncake_protocol,
                }
            },
        }
    else:
        kv_transfer_config = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path
            },
        }

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

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
