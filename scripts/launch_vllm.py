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


def _is_multimodal_config(config) -> bool:
    if any(
        hasattr(config, field_name)
        for field_name in (
            "vision_config",
            "visual_config",
            "image_token_id",
            "video_token_id",
        )
    ):
        return True

    model_type = str(getattr(config, "model_type", "")).lower()
    if any(
        marker in model_type
        for marker in ("vision", "llava", "mllama", "paligemma", "pixtral")
    ):
        return True

    architectures = getattr(config, "architectures", []) or []
    return any(
        any(
            marker in str(architecture).lower()
            for marker in (
                "vision",
                "llava",
                "mllama",
                "paligemma",
                "pixtral",
            )
        )
        for architecture in architectures
    )


def _ensure_hidden_state_extraction_defaults(cmd: list[str]) -> None:
    if not any(
        arg in cmd
        for arg in ("--enable-chunked-prefill", "--no-enable-chunked-prefill")
    ):
        cmd.append("--no-enable-chunked-prefill")

    # Prefix caching can make vLLM return full prompt IDs while the hidden-state
    # connector only writes uncached slots, so default extraction to full prompts.
    if not any(
        arg in cmd for arg in ("--enable-prefix-caching", "--no-enable-prefix-caching")
    ):
        cmd.append("--no-enable-prefix-caching")


def _normalize_target_layer_ids(
    target_layer_ids: list[int],
    *,
    num_hidden_layers: int,
    include_last_layer: bool,
) -> list[int]:
    """Return an unambiguous extraction order for custom verifier layers.

    Training always consumes the last extracted tensor as
    ``verifier_last_hidden_states``. Therefore an explicitly supplied final
    verifier layer must be unique and last, regardless of whether the user also
    requested automatic inclusion.
    """
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError(
            "--target-layer-ids must not contain duplicate layer IDs: "
            f"{target_layer_ids}"
        )

    out_of_range = [
        layer_id
        for layer_id in target_layer_ids
        if layer_id < 1 or layer_id > num_hidden_layers
    ]
    if out_of_range:
        raise ValueError(
            "--target-layer-ids must be in the inclusive range "
            f"[1, {num_hidden_layers}]; got {out_of_range}"
        )

    normalized = [
        layer_id for layer_id in target_layer_ids if layer_id != num_hidden_layers
    ]
    if not include_last_layer and num_hidden_layers not in target_layer_ids:
        raise ValueError(
            "--no-include-last-layer requires the final verifier layer "
            f"({num_hidden_layers}) to be listed explicitly in "
            "--target-layer-ids. Training consumes the last extracted tensor "
            "as verifier_last_hidden_states."
        )
    if include_last_layer or num_hidden_layers in target_layer_ids:
        normalized.append(num_hidden_layers)
    return normalized


def main():
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")

    from transformers import AutoConfig  # noqa: PLC0415

    from speculators.models.utils import (  # noqa: PLC0415
        default_auxiliary_target_layer_ids,
    )

    config = AutoConfig.from_pretrained(args.model)
    text_config = getattr(config, "text_config", None) or config
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError(
            "Model config must expose num_hidden_layers either at the top level "
            "or through a non-null text_config."
        )

    if _is_multimodal_config(config) and "--enforce-eager" not in vllm_args:
        warnings.warn(
            "Detected a multimodal verifier config. If your vLLM version has "
            "CUDA graph shape issues for multimodal hidden-state extraction, "
            "pass --enforce-eager explicitly after the launcher '--'.",
            stacklevel=2,
        )

    if args.target_layer_ids:
        target_layer_ids = _normalize_target_layer_ids(
            args.target_layer_ids,
            num_hidden_layers=num_hidden_layers,
            include_last_layer=args.include_last_layer,
        )
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly aligned in the training script. "
            "If the final verifier layer is included here, pass only the "
            "auxiliary layers to training.",
            stacklevel=2,
        )
    else:
        target_layer_ids = [
            *default_auxiliary_target_layer_ids(num_hidden_layers),
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
    hidden_states_path = os.path.abspath(os.path.expanduser(args.hidden_states_path))
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": hidden_states_path},
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

    _ensure_hidden_state_extraction_defaults(cmd)

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
