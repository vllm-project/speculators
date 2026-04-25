import argparse
import json
import os
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
    text_config = config.text_config if hasattr(config, "text_config") else config
    num_hidden_layers = text_config.num_hidden_layers

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly aligned in the training script. "
            "If the final verifier layer is included here, pass only the "
            "auxiliary layers to training.",
            stacklevel=2,
        )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]

    draft_hf_config = {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
    if text_config is not config:
        # vLLM's ExtractHiddenStatesConfig flattens the draft config and does not
        # preserve nested text_config for multimodal verifiers. Clear the nested
        # text_config and copy the text-only shape fields onto the draft config so
        # hidden-state extraction can derive the draft model shape from the
        # flattened config.
        draft_hf_config["text_config"] = None
        for field_name in (
            "num_attention_heads",
            "num_hidden_layers",
            "hidden_size",
            "num_key_value_heads",
            "head_dim",
        ):
            field_value = getattr(text_config, field_name, None)
            if field_value is not None:
                draft_hf_config[field_name] = field_value

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": draft_hf_config},
    }
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": args.hidden_states_path},
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

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
