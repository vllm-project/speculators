import argparse
import json
import os
import sys
import warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for hidden states extraction",
        usage=(
            "launch_vllm_nkv.py [-h] MODEL [--hidden-states-path HIDDEN_STATES_PATH] "
            "[--target-layer-ids ... | --attn-kv-layer-ids ... | --layer-mapping JSON] "
            "-- *VLLM_ARGS"
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
            "A (space separated) list of integer layer ids whose KV is extracted. "
            "Exactly one of --target-layer-ids / --attn-kv-layer-ids / --layer-mapping "
            "is required. If set, pass the same value into the training process."
        ),
    )
    parser.add_argument(
        "--attn-kv-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "(mamba_attn_hybrid) Verifier KV layer ids the hybrid's attention blocks "
            "read; the data's KV set is sorted(set(...)). Mutually exclusive with "
            "--target-layer-ids / --layer-mapping. Pass the same ids to "
            "scripts/train.py --attn-kv-layer-ids."
        ),
    )
    parser.add_argument(
        "--layer-mapping",
        type=str,
        default=None,
        help=(
            "JSON list-of-lists mapping draft layers to verifier layers. "
            "e.g. '[[1,2],[9,10],[17,18],[25,26],[33,34]]'. "
            "Single-element groups use KV as-is, multi-element groups average. "
            "Mutually exclusive with --target-layer-ids."
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
        "--latent-fusion-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "(mamba_attn_hybrid: REQUIRED) Verifier layer ids whose "
            "hidden states are saved "
            "as the SSM latent-fusion seed (vLLM stores them in ascending layer-index "
            "order, before the final layer). If unset, only the final layer is saved "
            "(the default) and a mamba_attn_hybrid training run fails at the first "
            "forward with an fc width mismatch. Pass the SAME ids you pass to "
            "scripts/train.py --latent-fusion-layer-ids (which defaults to "
            "--target-layer-ids when unset)."
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

    if not any(
        "enable-prefix-caching" in a or "enable_prefix_caching" in a for a in vllm_args
    ):
        vllm_args.append("--no-enable-prefix-caching")
    if not any(
        "enable-chunked-prefill" in a or "enable_chunked_prefill" in a
        for a in vllm_args
    ):
        vllm_args.append("--no-enable-chunked-prefill")

    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config
    num_hidden_layers = config.num_hidden_layers

    layer_mapping = None
    if args.layer_mapping is not None:
        layer_mapping = json.loads(args.layer_mapping)
        if not isinstance(layer_mapping, list) or not all(
            isinstance(g, list) and all(isinstance(x, int) for x in g)
            for g in layer_mapping
        ):
            raise ValueError("--layer-mapping must be a JSON list of lists of ints")

    n_kv_sources = sum(
        x is not None
        for x in (args.target_layer_ids, args.attn_kv_layer_ids, layer_mapping)
    )
    if n_kv_sources != 1:
        raise ValueError(
            "Exactly one of --target-layer-ids, --attn-kv-layer-ids, or "
            "--layer-mapping must be provided"
        )

    if layer_mapping is not None:
        target_layer_ids = sorted({lid for group in layer_mapping for lid in group})
    elif args.attn_kv_layer_ids is not None:
        target_layer_ids = sorted(set(args.attn_kv_layer_ids))
        warnings.warn(
            f"Extracting KV for attn-kv layers {target_layer_ids}. Pass the SAME ids "
            "into scripts/train.py --attn-kv-layer-ids.",
            stacklevel=2,
        )
    else:
        target_layer_ids = args.target_layer_ids
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly passed into the training script.",
            stacklevel=2,
        )

    if args.attn_kv_layer_ids is not None and args.latent_fusion_layer_ids is None:
        raise ValueError(
            "--latent-fusion-layer-ids is required with --attn-kv-layer-ids "
            "(mamba_attn_hybrid uses the fused verifier hidden as the SSM seed). "
            "Pass the same ids as scripts/train.py --latent-fusion-layer-ids."
        )

    fusion_ids = list(args.latent_fusion_layer_ids or [])
    if num_hidden_layers in fusion_ids:
        raise ValueError(
            f"--latent-fusion-layer-ids must not include the final layer "
            f"{num_hidden_layers}; it is appended automatically as the target layer."
        )
    hidden_state_layer_ids = list(fusion_ids)
    if args.include_last_layer:
        hidden_state_layer_ids.append(num_hidden_layers)

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": hidden_state_layer_ids}
        },
    }
    extra_config = {
        "shared_storage_path": args.hidden_states_path,
        "target_layer_ids": target_layer_ids,
    }
    if layer_mapping is not None:
        extra_config["layer_mapping"] = layer_mapping
    kv_transfer_config = {
        "kv_connector": "NKVDataConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "speculators.data_generation.nkv_connector",
        "kv_connector_extra_config": extra_config,
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
