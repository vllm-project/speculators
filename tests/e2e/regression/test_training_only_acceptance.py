import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from tests.e2e.utils import run_training, run_vllm_engine
from tests.utils import requires_cadence

_CONFIGS_DIR = Path(__file__).parent / "configs" / "training"


def _load_configs() -> list[dict]:
    configs = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            configs.append(json.load(f))
    return configs


_CONFIGS = _load_configs()


def _resolve_repo(repo_id: str, repo_type: str = "dataset") -> Path:
    """Return a local Path for a repo, downloading from HuggingFace if needed.

    Tries the local cache first (no network call) and falls back to a full
    download only when the data is not already cached.
    """
    path = Path(repo_id)
    if path.is_absolute() or repo_id.startswith(("./", "../")):
        return path
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id, repo_type=repo_type, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type))


@requires_cadence("nightly")
@pytest.mark.regression
@pytest.mark.parametrize("config", _CONFIGS, ids=[c["name"] for c in _CONFIGS])
def test_training_acceptance(
    config: dict[str, Any], tmp_path: Path, prompts: list[list[dict[str, str]]]
):
    save_path = tmp_path / "checkpoints"

    # 1. Fetch precomputed hidden states
    hidden_states_dir = _resolve_repo(config["hidden_states_repo"])

    # 2. Set arg to build for vocab mapping from token_freq.pt in dataset
    extra_train_args = ["--draft-vocab-size", str(config["draft_vocab_size"])]

    # 3. Run training
    training_cfg = config["training"]
    run_training(
        model=config["verifier_model"],
        data_path=hidden_states_dir,
        save_path=save_path,
        seq_length=training_cfg["total_seq_len"],
        epochs=training_cfg["epochs"],
        lr=training_cfg["lr"],
        online=False,
        extra_train_args=extra_train_args or None,
    )

    # 4. Validate trained model meets acceptance thresholds in vLLM
    epochs = training_cfg["epochs"]
    final_checkpoint = str(save_path / str(epochs - 1))
    run_vllm_engine(
        model_path=final_checkpoint,
        tmp_path=tmp_path,
        max_tokens=512,
        ignore_eos=True,
        prompts=prompts,
        acceptance_thresholds=config["acceptance_thresholds"],
    )
