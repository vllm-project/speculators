"""Exercise the B16 example's CLI without starting GPU training."""

import os
import subprocess
from pathlib import Path

from speculators.train.config import TrainConfig


def test_dpard_example_resolves_native_b16_config(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    launcher = tmp_path / "torchrun"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o755)
    data = tmp_path / "native data"
    output = tmp_path / "training output"
    result = subprocess.run(  # noqa: S603 -- fixed repository script; GPU launcher stubbed
        ["/bin/bash", str(repo / "examples/train/dspark_qwen3_4b_dpard_offline.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DATA_PATH": str(data),
            "OUTPUT_DIR": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    cfg = TrainConfig.resolve(argv[argv.index("scripts/train.py") + 1 :])
    assert cfg.speculator_type == "dspark"
    assert cfg.data.data_path == str(data)
    assert cfg.trainer.save_path == str(output)
    assert cfg.dflash.block_size == 16
    assert cfg.dflash.sample_from_anchor is True
    assert cfg.data.max_anchors == 512
    assert cfg.draft.num_layers == 3
    assert cfg.draft.target_layer_ids == [1, 17, 33]
    assert cfg.draft.full_attention_indices == []
    assert cfg.draft.sliding_window == 2048
    assert cfg.draft.sliding_window_non_causal is False
    assert cfg.loss.loss_fn == "renyi_half"
    assert cfg.dflash.per_position_loss_weight == "dpard"
    assert cfg.dspark.dpard_alpha == 0.5
    assert cfg.dflash.dflash_decay_gamma == 7.0
    assert cfg.dspark.enable_confidence_head is True
    assert cfg.optimizer.optimizer == "adamw"
    assert cfg.optimizer.lr == 6e-4
    assert cfg.optimizer.weight_decay == 0.01
    assert cfg.scheduler.scheduler_type == "linear"
    assert cfg.scheduler.scheduler_warmup_ratio == 0.04
    assert cfg.seed == 42
    assert cfg.trainer.epochs == 6
    assert cfg.data.total_seq_len == 8192
    assert cfg.data.noise_std == 0.05
    assert cfg.generation.on_missing == "raise"
