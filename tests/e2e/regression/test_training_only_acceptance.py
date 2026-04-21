from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from tests.e2e.utils import run_training, run_vllm_engine
from tests.utils import requires_cadence


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


@requires_cadence("weekly")
@pytest.mark.regression
def test_eagle3_qwen3_8b_sharegpt(tmp_path: Path, prompts: list[list[dict[str, str]]]):
    save_path = tmp_path / "checkpoints"
    hidden_states_dir = _resolve_repo("inference-optimization/Qwen3-8b-sharegpt-5k")
    epochs = 5
    run_training(
        model="Qwen/Qwen3-8B",
        speculator_type="eagle3",
        data_path=hidden_states_dir,
        save_path=save_path,
        seq_length=8192,
        epochs=epochs,
        lr=3e-4,
        draft_vocab_size=8192,
        online=False,
    )
    final_checkpoint = str(save_path / str(epochs - 1))
    run_vllm_engine(
        model_path=final_checkpoint,
        tmp_path=tmp_path,
        max_tokens=512,
        ignore_eos=True,
        prompts=prompts,
        acceptance_thresholds=[0.40, 0.10, 0.01],
    )


@requires_cadence("weekly")
@pytest.mark.regression
def test_dflash_qwen3_8b_sharegpt(tmp_path: Path, prompts: list[list[dict[str, str]]]):
    save_path = tmp_path / "checkpoints"
    hidden_states_dir = _resolve_repo("inference-optimization/Qwen3-8b-sharegpt-5k")
    epochs = 5
    run_training(
        model="Qwen/Qwen3-8b",
        speculator_type="dflash",
        data_path=hidden_states_dir,
        save_path=save_path,
        seq_length=8192,
        epochs=epochs,
        lr=3e-4,
        draft_vocab_size=8192,
        num_layers=3,
        online=False,
    )
    final_checkpoint = str(save_path / str(epochs - 1))
    run_vllm_engine(
        model_path=final_checkpoint,
        tmp_path=tmp_path,
        max_tokens=512,
        ignore_eos=True,
        prompts=prompts,
        acceptance_thresholds=[0.30, 0.05, 0.001, 0, 0, 0, 0],
    )
