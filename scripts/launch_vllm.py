import argparse
import json
import os
import sys
import warnings


def unwrap_verifier_configs(config):
    """Return multimodal/container config and the text backbone config."""
    multimodal_config = getattr(config, "thinker_config", config)
    text_config = multimodal_config
    if hasattr(text_config, "text_config"):
        text_config = text_config.text_config
    return multimodal_config, text_config


def get_deepstack_visual_indexes(multimodal_config) -> list[int]:
    """Get DeepStack layer indexes when present on the verifier."""
    vision_config = getattr(multimodal_config, "vision_config", None)
    deepstack_layers = getattr(vision_config, "deepstack_visual_indexes", None)
    if deepstack_layers is None:
        deepstack_layers = getattr(multimodal_config, "deepstack_visual_indexes", None)
    return list(deepstack_layers or [])


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
            "For DFlash models, append the last layer (num_hidden_layers) to "
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

    raw_config = AutoConfig.from_pretrained(args.model)
    multimodal_config, config = unwrap_verifier_configs(raw_config)
    num_hidden_layers = config.num_hidden_layers

    if args.target_layer_ids:
        target_layer_ids = list(args.target_layer_ids)
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly passed into the training script.",
            stacklevel=2,
        )
    else:
        deepstack_layers = set(get_deepstack_visual_indexes(multimodal_config))
        candidate_layer_ids = [2, num_hidden_layers // 2, num_hidden_layers - 3]
        target_layer_ids = [
            layer_id - 1 if layer_id in deepstack_layers else layer_id
            for layer_id in candidate_layer_ids
        ]
        if args.include_last_layer:
            target_layer_ids.append(num_hidden_layers)

    # Build hf_config overrides for ExtractHiddenStatesConfig.
    #
    # vLLM's SpeculativeConfig("extract_hidden_states") builds a draft hf_config
    # via ExtractHiddenStatesConfig(target_hf_config, **hf_config_overrides) and
    # merges them as `{**target_hf_config.to_dict(), **kwargs}` (kwargs win).
    #
    # For composite multimodal configs (e.g. Qwen3-Omni*, Qwen3-VL-MoE),
    # `target_hf_config.to_dict()` keeps the text backbone either under nested
    # containers such as `thinker_config -> text_config` or under a top-level
    # selector attr like `text_config`. In both cases, the rebuilt
    # ExtractHiddenStatesConfig can fail vLLM's `get_hf_text_config()` path:
    # either the text attrs are no longer directly visible on the wrapper, or a
    # stale nested dict is returned instead of a real config-like object.
    # That leads to the "text_config extracted ... does not have
    # `num_attention_heads` attribute" ValidationError.
    #
    # Fix: flatten the text backbone's fields into the hf_config override.
    # Since kwargs override the flattened model_dict in ExtractHiddenStatesConfig,
    # the resulting draft hf_config will expose text-backbone attributes at the
    # top level (num_attention_heads, hidden_size, num_hidden_layers,
    # vocab_size, ...), which is what vLLM's get_hf_text_config() requires.
    hf_config_overrides: dict = {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
    if config is not raw_config:
        # Nested multimodal target: promote text-backbone attrs to top level.
        # `to_dict()` recursively serialises sub-configs; we strip fields that
        # would conflict with ExtractHiddenStatesConfig's forced
        # `architectures`/`model_type`.
        _text_cfg_dict = config.to_dict()
        for _k in ("architectures", "model_type", "auto_map", "torch_dtype"):
            _text_cfg_dict.pop(_k, None)

        # Some multimodal parent configs (e.g. Qwen3-VL-MoE) expose the text
        # backbone via selector attrs such as `text_config`. When vLLM rebuilds
        # the draft config as ExtractHiddenStatesConfig(parent_hf_config,
        # **hf_config_overrides), those parent attrs survive as plain dicts.
        # Then `PretrainedConfig.get_text_config()` returns the stale nested
        # dict instead of falling back to the promoted top-level text attrs,
        # and `get_hf_text_config()` raises because dict values do not expose
        # attribute access like `num_attention_heads`.
        #
        # Explicitly neutralise all selector attrs that HF may probe so the
        # rebuilt config falls back to the promoted top-level text fields.
        for _nested_text_attr in (
            "text_config",
            "text_encoder",
            "decoder",
            "generator",
        ):
            _text_cfg_dict[_nested_text_attr] = None

        # kwargs win over model_dict in ExtractHiddenStatesConfig; this surfaces
        # `num_attention_heads` et al. at top level so get_hf_text_config()
        # returns a valid text config.
        hf_config_overrides = {**_text_cfg_dict, **hf_config_overrides}

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": hf_config_overrides},
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
