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


def deduplicate_layer_ids(layer_ids: list[int]) -> list[int]:
    """Deduplicate layer ids while preserving user-specified order."""
    seen: set[int] = set()
    result: list[int] = []
    for layer_id in layer_ids:
        if layer_id not in seen:
            seen.add(layer_id)
            result.append(layer_id)
    return result


def validate_layer_ids(
    layer_ids: list[int], num_hidden_layers: int, option_name: str
) -> list[int]:
    """Validate vLLM post-layer ids and preserve the effective order."""
    validated = deduplicate_layer_ids(layer_ids)
    invalid = [
        layer_id
        for layer_id in validated
        if layer_id < 0 or layer_id > num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"{option_name} contains invalid layer ids {invalid}. "
            f"Expected ids in [0, {num_hidden_layers}], where 0 is the "
            "embedding output and num_hidden_layers is the final decoder output."
        )
    return validated


def get_default_target_layer_ids(multimodal_config, num_hidden_layers: int) -> list[int]:
    """Return default auxiliary layer ids used by training."""
    deepstack_layers = set(get_deepstack_visual_indexes(multimodal_config))
    candidate_layer_ids = [2, num_hidden_layers // 2, num_hidden_layers - 3]
    return [
        layer_id - 1 if layer_id in deepstack_layers else layer_id
        for layer_id in candidate_layer_ids
    ]


def resolve_layer_ids(args, multimodal_config, num_hidden_layers: int):
    """Resolve training layer ids and exact vLLM extraction layer ids.

    vLLM's extract_hidden_states path stores exactly the configured
    eagle_aux_hidden_state_layer_ids. DFlash data loading consumes all but the
    last stored slice as training auxiliary hidden states and treats the last
    slice as the verifier/final reference state. Therefore, when
    --include-last-layer is enabled, the final layer is forced to the end of the
    extraction list but is not reported as a training target layer id.
    """
    if args.target_layer_ids:
        cli_layer_ids = validate_layer_ids(
            list(args.target_layer_ids), num_hidden_layers, "--target-layer-ids"
        )
        # If users pass the final layer explicitly, keep DFlash layout correct by
        # moving it to the last extracted slice. This avoids a costly runtime
        # tensor reorder and preserves the loader's [:, :-1] / [:, -1] split.
        target_layer_ids = [
            layer_id for layer_id in cli_layer_ids if layer_id != num_hidden_layers
        ]
        if not target_layer_ids and args.include_last_layer:
            raise ValueError(
                "--target-layer-ids must contain at least one non-final auxiliary "
                "layer when --include-last-layer is enabled."
            )
        source = "custom"
    else:
        target_layer_ids = validate_layer_ids(
            get_default_target_layer_ids(multimodal_config, num_hidden_layers),
            num_hidden_layers,
            "default target layer ids",
        )
        source = "default"

    extraction_layer_ids = list(target_layer_ids)
    if args.include_last_layer:
        extraction_layer_ids.append(num_hidden_layers)
    extraction_layer_ids = validate_layer_ids(
        extraction_layer_ids, num_hidden_layers, "resolved extraction layer ids"
    )
    if not extraction_layer_ids:
        raise ValueError(
            "At least one vLLM extraction layer id must be selected. Pass one or "
            "more --target-layer-ids values or keep --include-last-layer enabled."
        )

    return target_layer_ids, extraction_layer_ids, source


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
        "--layer-ids",
        dest="target_layer_ids",
        type=int,
        nargs="+",
        help=(
            "Auxiliary post-layer ids to extract for training. Alias: --layer-ids. "
            "vLLM layer ids are in [0, num_hidden_layers], where 0 is the "
            "embedding output and num_hidden_layers is the final decoder output. "
            "When --include-last-layer is enabled, num_hidden_layers is appended "
            "to the vLLM extraction ids and should not be passed to training. "
            "Defaults to [2, num_hidden_layers // 2, num_hidden_layers - 3]."
        ),
    )
    parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For DFlash models, append the last layer (num_hidden_layers) to the "
            "vLLM extraction ids as the final verifier/reference slice. "
            "Default: True"
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
        training_target_layer_ids, extraction_layer_ids, layer_id_source = resolve_layer_ids(
            args, multimodal_config, num_hidden_layers
        )
        warnings.warn(
            "Using custom target layer ids. Pass "
            f"{training_target_layer_ids} to the training script; vLLM will "
            f"extract {extraction_layer_ids}.",
            stacklevel=2,
        )
    else:
        training_target_layer_ids, extraction_layer_ids, layer_id_source = resolve_layer_ids(
            args, multimodal_config, num_hidden_layers
        )

    print(
        "Layer ids: "
        f"source={layer_id_source}, training_target_layer_ids="
        f"{training_target_layer_ids}, extraction_layer_ids={extraction_layer_ids}"
    )

    # Build overrides for ExtractHiddenStatesConfig.
    # For nested multimodal configs, promote text-backbone fields to top level
    # so vLLM can resolve a valid text config.
    hf_config_overrides: dict = {"eagle_aux_hidden_state_layer_ids": extraction_layer_ids}
    if config is not raw_config:
        # Promote nested text-backbone fields; drop conflicting wrapper fields.
        _text_cfg_dict = config.to_dict()
        for _k in ("architectures", "model_type", "auto_map", "torch_dtype"):
            _text_cfg_dict.pop(_k, None)

        # Clear nested selector attrs so HF falls back to promoted top-level
        # text fields instead of stale dict payloads.
        for _nested_text_attr in (
            "text_config",
            "text_encoder",
            "decoder",
            "generator",
        ):
            _text_cfg_dict[_nested_text_attr] = None

        # kwargs override model_dict, exposing text attrs at top level.
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

    disable_cp_arg = "--no-enable-chunked-prefill"
    if disable_cp_arg not in cmd:
        cmd.append(disable_cp_arg)

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
