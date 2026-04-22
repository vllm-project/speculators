import sys
from pathlib import Path

# Add scripts directory to path so we can import the run_e2e function.
scripts_path = Path(__file__).absolute().parent.parent.parent / "scripts"
sys.path.append(str(scripts_path))

from gen_and_train import (  # noqa: E402
    DataGenArgs,
    TrainArgs,
    VocabMappingArgs,
    run_e2e,
)

### Example E2E full run for Qwen3-Omni-Thinking on ShareGPT ###

# This example trains a 1-layer dense Qwen3 DFlash draft model against the
# Qwen3-Omni thinker text tower using text-only ShareGPT data.
#
# The vLLM hidden-state extraction step captures 3 auxiliary layers plus the
# verifier final layer. During training, only the auxiliary layers are passed as
# --target-layer-ids because the final verifier layer is stored separately in the
# current training data format as verifier_last_hidden_states.

if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    OUTPUT_PATH = "./output/qwen3_omni_thinking_sharegpt"
    TOTAL_SEQ_LEN = 8192

    AUX_TARGET_LAYER_IDS = [2, 23, 45]
    CAPTURE_LAYER_IDS = [*AUX_TARGET_LAYER_IDS, 48]
    DRAFT_VOCAB_SIZE = 32000

    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        seq_length=TOTAL_SEQ_LEN,
        turn_dropout=True,
        layer_ids=CAPTURE_LAYER_IDS,
    )

    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        target_vocab_size=152064,
    )

    train_args = TrainArgs(
        logger="trackio",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="qwen3_omni_thinking_sharegpt",
        epochs=10,
        speculator_type="dflash",
        draft_arch="qwen3",
        num_layers=1,
        draft_intermediate_size=6144,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        target_layer_ids=AUX_TARGET_LAYER_IDS,
        mask_token_id=151671,
        block_size=8,
        max_anchors=256,
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=data_gen_args_sharegpt,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )
