"""End-to-end runner for DFlash training on Qwen3-Omni-Thinking + custombench.

Every knob that used to be hard-coded in the smoke test is now a CLI flag.
No relative paths are assumed; the JSONL, verifier checkpoint, and output
directory must all be passed explicitly (absolute paths recommended).

Pipeline:
  1. ``prepare_data.py --multimodal`` tokenizes the JSONL, expands <video>
     placeholders via AutoProcessor, and dumps multimodal sidecars.
  2. ``data_generation_offline2.py`` queries a vLLM-served
     Qwen3-Omni-Thinking for per-layer hidden states on each chat-formatted
     multimodal message.
  3. ``train.py --multimodal`` loads the sidecars, reconstructs 3D MRoPE
     position_ids via the thinker's ``get_rope_index``, and trains the DFlash
     draft model.

Prerequisite (produce the JSONL first):
    python build_custombench_jsonl.py \
        --csv          <csv with abo_zh_caption column> \
        --dataset-path <video dir> \
        --out          <abs path>/custombench_train.jsonl \
        --max-samples  5 --require-video-exists

Smoke-test example (5 samples, 1 epoch, 1 TTT step):
    python qwen3_omni_thinking_custombench.py \
        --verifier-path /home/ray/model_ckpt \
        --train-jsonl   /abs/path/custombench_smoke.jsonl \
        --output-path   /abs/path/output/qwen3_omni_smoke \
        --max-samples 5 --epochs 1 --ttt-steps 1

Production example:
    python qwen3_omni_thinking_custombench.py \
        --verifier-path /models/Qwen3-Omni-30B-A3B-Thinking \
        --train-jsonl   /data/custombench_train.jsonl \
        --output-path   /runs/qwen3_omni_dflash \
        --max-samples 50000 --epochs 3 --ttt-steps 3 --lr 3e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make gen_and_train importable regardless of CWD.
_SCRIPTS_DIR = Path(__file__).absolute().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_SCRIPTS_DIR))

from gen_and_train import (  # noqa: E402
    DataGenArgs,
    TrainArgs,
    VocabMappingArgs,
    run_e2e,
)


def _parse_int_list(raw: str) -> list[int]:
    """Accept '2,23,45' or '2 23 45' for layer-id style flags."""
    if not raw:
        return []
    parts = raw.replace(",", " ").split()
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected space/comma-separated integers, got: {raw!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ---- Required paths ------------------------------------------------------
    p.add_argument(
        "--verifier-path",
        required=True,
        type=Path,
        help="Absolute path to the verifier (Qwen3-Omni-Thinking) checkpoint.",
    )
    p.add_argument(
        "--train-jsonl",
        required=True,
        type=Path,
        help=(
            "Absolute path to the JSONL produced by build_custombench_jsonl.py. "
            "Triggers the datasets 'json' branch inside load_raw_dataset()."
        ),
    )
    p.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Absolute root directory for gen/ vocab_mapping/ checkpoints/ logs/.",
    )

    # ---- Data-generation knobs ----------------------------------------------
    p.add_argument(
        "--dataset-name",
        default="custombench",
        help=(
            "Required by _infer_dataset_name() when train-jsonl is a local file; "
            "only sharegpt/ultrachat/llava-instruct are auto-inferred."
        ),
    )
    p.add_argument("--total-seq-len", type=int, default=16384)
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=2,
        help="CPU workers for chat-template tokenization.",
    )
    p.add_argument(
        "--turn-dropout",
        action="store_true",
        help="Random first-N turns augmentation; off by default for small runs.",
    )
    p.add_argument(
        "--aux-target-layer-ids",
        type=_parse_int_list,
        default=[2, 23, 45],
        help=(
            "Auxiliary hidden-state layers captured by DFlash draft input. "
            "Must equal the non-last entries in --capture-layer-ids."
        ),
    )
    p.add_argument(
        "--last-layer-id",
        type=int,
        default=48,
        help=(
            "num_hidden_layers of the verifier's text backbone. For "
            "Qwen3-Omni-Thinking this is 48 (thinker.text_config)."
        ),
    )

    # ---- Vocab mapping -------------------------------------------------------
    p.add_argument(
        "--draft-vocab-size",
        type=int,
        default=32000,
        help="Draft-side frequent-token vocabulary size.",
    )
    p.add_argument(
        "--target-vocab-size",
        type=int,
        default=None,
        help=(
            "Verifier vocab size. If omitted (default), it is auto-inferred "
            "from the verifier config via the same unwrap path as train.py "
            "(thinker_config -> text_config -> .vocab_size). Only override "
            "if you truly need to force a value."
        ),
    )

    # ---- Training hyperparams -----------------------------------------------
    p.add_argument("--run-name", default="qwen3_omni_thinking_custombench")
    p.add_argument("--logger", default="trackio", choices=("trackio", "wandb", "none"))
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ttt-steps", type=int, default=1)
    p.add_argument("--speculator-type", default="dflash")
    p.add_argument("--draft-arch", default="qwen3")
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--draft-intermediate-size", type=int, default=6144)
    p.add_argument(
        "--mask-token-id",
        type=int,
        default=151671,
        help=(
            "Token id used as DFlash's masked-diffusion placeholder. "
            "Must be a dedicated special token in the verifier tokenizer."
        ),
    )
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--max-anchors", type=int, default=256)

    return p.parse_args()


def _resolve_target_vocab_size(verifier_path: Path, override: int | None) -> int:
    """Pick the integer that both prepare_data and train.py will agree on.

    train.py::unwrap_verifier_text_config does:
        thinker_config -> text_config -> .vocab_size
    We replicate it *exactly* so the t2d.npy we cache has
    ``shape[0] == transformer_layer_config.vocab_size`` at train time.
    Otherwise ``DraftVocabMixin.load_vocab_mappings`` raises:
        t2d.shape[0] (X) must match verifier_vocab_size (Y).
    """
    if override is not None:
        return override

    # Local import keeps CLI --help snappy (no transformers import on --help).
    from transformers import AutoConfig  # noqa: PLC0415

    cfg = AutoConfig.from_pretrained(str(verifier_path), trust_remote_code=True)
    if hasattr(cfg, "thinker_config"):
        cfg = cfg.thinker_config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    vocab = getattr(cfg, "vocab_size", None)
    if not isinstance(vocab, int):
        raise SystemExit(
            "Could not auto-detect verifier text vocab_size from "
            f"{verifier_path}. Pass --target-vocab-size explicitly."
        )
    return vocab


def _build_args(ns: argparse.Namespace) -> tuple[DataGenArgs, VocabMappingArgs, TrainArgs]:
    # Enforce that train-jsonl + verifier-path are absolute; CLI should not
    # silently depend on caller CWD for either resource.
    train_jsonl = ns.train_jsonl.expanduser().resolve()
    if not train_jsonl.is_file():
        raise SystemExit(f"--train-jsonl does not exist: {train_jsonl}")
    verifier_path = ns.verifier_path.expanduser().resolve()
    if not verifier_path.exists():
        raise SystemExit(f"--verifier-path does not exist: {verifier_path}")

    capture_layer_ids = [*ns.aux_target_layer_ids, ns.last_layer_id]

    resolved_target_vocab = _resolve_target_vocab_size(
        verifier_path, ns.target_vocab_size
    )

    data_gen_args = DataGenArgs(
        train_data_path=str(train_jsonl),
        dataset_name=ns.dataset_name,
        seq_length=ns.total_seq_len,
        turn_dropout=ns.turn_dropout,
        multimodal=True,
        layer_ids=capture_layer_ids,
        max_samples=ns.max_samples,
        num_preprocessing_workers=ns.num_preprocessing_workers,
    )

    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=ns.draft_vocab_size,
        target_vocab_size=resolved_target_vocab,
    )

    train_args = TrainArgs(
        logger=ns.logger,
        lr=ns.lr,
        total_seq_len=ns.total_seq_len,
        run_name=ns.run_name,
        epochs=ns.epochs,
        ttt_steps=ns.ttt_steps,
        speculator_type=ns.speculator_type,
        draft_arch=ns.draft_arch,
        num_layers=ns.num_layers,
        draft_intermediate_size=ns.draft_intermediate_size,
        draft_vocab_size=ns.draft_vocab_size,
        target_layer_ids=ns.aux_target_layer_ids,
        mask_token_id=ns.mask_token_id,
        block_size=ns.block_size,
        max_anchors=ns.max_anchors,
    )

    return data_gen_args, vocab_mapping_args, train_args


def main() -> None:
    ns = parse_args()
    data_gen_args, vocab_mapping_args, train_args = _build_args(ns)

    output_path = ns.output_path.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    run_e2e(
        verifier_name_or_path=str(ns.verifier_path.expanduser().resolve()),
        output_path=str(output_path),
        data_gen_args=data_gen_args,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )


if __name__ == "__main__":
    main()
