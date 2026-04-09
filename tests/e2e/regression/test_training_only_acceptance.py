import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution
from tests.e2e.utils import run_vllm_engine

_CONFIGS_DIR = Path(__file__).parent / "configs" / "training"


def _load_configs() -> list[dict]:
    configs = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            configs.append(json.load(f))
    return configs


_CONFIGS = _load_configs()


def _build_vocab_mappings(
    token_freq_dir: Path,
    d2t_path: Path,
    t2d_path: Path,
    draft_vocab_size: int,
    target_vocab_size: int,
):
    token_freq_dict = torch.load(
        token_freq_dir / "token_freq_sharegpt.pt", weights_only=True
    )
    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=draft_vocab_size,
        target_vocab_size=target_vocab_size,
    )
    np.save(d2t_path, d2t.cpu().numpy())
    np.save(t2d_path, t2d.cpu().numpy())


def _resolve_repo(repo_id: str, repo_type: str = "dataset") -> Path:
    """Return a local Path for a repo, downloading from HuggingFace if needed."""
    path = Path(repo_id)
    if path.is_absolute() or repo_id.startswith(("./", "../")):
        return path
    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type))


def _resolve_vocab_mappings(config: dict, tmp_path: Path) -> tuple[Path, Path]:
    """Return (d2t_path, t2d_path).

    If the config provides a ``vocab_mapping_repo``, the pre-computed files are
    downloaded directly.  Otherwise they are built from the token-frequency
    distribution in ``token_freq_repo``.
    """
    d2t_path = tmp_path / "d2t.npy"
    t2d_path = tmp_path / "t2d.npy"

    if "vocab_mapping_repo" in config:
        mapping_dir = _resolve_repo(config["vocab_mapping_repo"])
        shutil.copy(mapping_dir / "d2t.npy", d2t_path)
        shutil.copy(mapping_dir / "t2d.npy", t2d_path)
    else:
        token_freq_dir = _resolve_repo(config["token_freq_repo"])
        _build_vocab_mappings(
            token_freq_dir=token_freq_dir,
            d2t_path=d2t_path,
            t2d_path=t2d_path,
            draft_vocab_size=config["draft_vocab_size"],
            target_vocab_size=config["target_vocab_size"],
        )

    return d2t_path, t2d_path


def _run_training(args: dict) -> subprocess.CompletedProcess:
    cmd = ["python", "scripts/train.py"]
    for key, value in args.items():
        flag = f"--{key}"
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    logger.info("Training command: {}", " ".join(cmd))
    return subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.regression
@pytest.mark.parametrize("config", _CONFIGS, ids=[c["name"] for c in _CONFIGS])
def test_training_acceptance(
    config: dict, tmp_path: Path, prompts: list[list[dict[str:str]]]
):
    save_path = tmp_path / "checkpoints"

    # 1. Fetch precomputed hidden states
    hidden_states_dir = _resolve_repo(config["hidden_states_repo"])

    # 2. Resolve vocab mappings (pre-computed or built from token frequencies)
    d2t_path, t2d_path = _resolve_vocab_mappings(config, tmp_path)

    # 3. Run training
    training_cfg = config["training"]
    training_args = {
        "verifier-name-or-path": config["verifier_model"],
        "data-path": hidden_states_dir,
        "save-path": str(save_path),
        "log-dir": str(tmp_path / "logs"),
        "d2t-path": str(d2t_path),
        "t2d-path": str(t2d_path),
        "lr": training_cfg["lr"],
        "total-seq-len": training_cfg["total_seq_len"],
        "epochs": training_cfg["epochs"],
    }
    if training_cfg.get("legacy_data"):
        training_args["legacy-data"] = True

    result = _run_training(training_args)

    if result.returncode != 0:
        logger.error("Training stdout:\n{}", result.stdout)
        logger.error("Training stderr:\n{}", result.stderr)
    assert result.returncode == 0, "Training subprocess failed"

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
