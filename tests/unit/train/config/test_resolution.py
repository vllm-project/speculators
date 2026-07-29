"""Resolution seam tests plus the example-recipe back-compat suite.

The recipe suite is the central backward-compatibility guard: each real
``examples/train/*.sh`` recipe's ``train.py`` invocation is extracted (bash does
the variable expansion and word-splitting, exactly as a user's shell would) and
run through :meth:`TrainConfig.resolve`; the flags it sets must resolve to their
expected values in the flat dict. A renamed, dropped, or retyped flag breaks a
real recipe. There is deliberately no golden ``vars(args)`` snapshot.
"""

import re
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from speculators.train.config import TrainConfig
from speculators.train.config.resolution import (
    ConfigError,
    build_parser,
)
from speculators.train.config.schema import nest_flat

EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "train"


# --------------------------------------------------------------------------- #
# from_sources: the pure, argv-free core
# --------------------------------------------------------------------------- #


def test_from_sources_is_pure_and_flag_beats_default():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "some-model", "lr": 0.5}, argv=["train.py"]
    )
    flat = cfg.flatten()
    assert flat["verifier_name_or_path"] == "some-model"
    assert flat["lr"] == 0.5  # flag wins
    assert flat["epochs"] == 20  # untouched -> schema default


def test_from_sources_records_winning_layer_in_memory():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "lr": 0.5}, argv=["train.py"]
    )
    assert cfg._provenance["lr"] == "flag"
    assert cfg._provenance["epochs"] == "default"


def test_from_sources_raises_instead_of_exiting():
    # checkpoint_freq=1.5 fails the schema validator (>1 must be a whole number).
    with pytest.raises(ValidationError):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m", "checkpoint_freq": 1.5},
            argv=["train.py"],
        )


def test_from_sources_rejects_float16_dtype():
    # float16 needs gradient scaling, which the trainer does not implement; the
    # dtype validator rejects it at config-construction time.
    with pytest.raises(ValidationError):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m", "hidden_states_dtype": "float16"},
            argv=["train.py"],
        )


def test_from_sources_rejects_non_dtype_torch_attr():
    # hasattr(torch, "nn") is True but it is not a dtype; the validator must
    # reject it at config time rather than let it fail later as an opaque
    # autocast error.
    with pytest.raises(ValidationError):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m", "hidden_states_dtype": "nn"},
            argv=["train.py"],
        )


def test_from_sources_missing_required_names_the_flag():
    # The verifier is required, but the requirement is enforced post-build (so a
    # --config can supply it), naming the flag when nothing does.
    with pytest.raises(ConfigError, match="--verifier-name-or-path"):
        TrainConfig.from_sources(cli={}, argv=["train.py"])


def test_from_sources_rejects_draft_init_conflict():
    # The conflict is decided from the in-memory winning-layer record: num_layers
    # won by the flag layer, so it conflicts with --from-pretrained.
    with pytest.raises(ConfigError, match="from-pretrained"):
        TrainConfig.from_sources(
            cli={
                "verifier_name_or_path": "m",
                "from_pretrained": "ckpt",
                "num_layers": 2,
            },
            argv=["train.py"],
        )


# --------------------------------------------------------------------------- #
# from_sources: the YAML layer and full flag > yaml > default precedence
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return str(path)


def test_yaml_beats_default_and_flag_beats_yaml(tmp_path):
    # lr: set in all three layers -> flag wins; epochs: only YAML -> YAML beats
    # default; weight_decay: nowhere -> schema default.
    config_path = _write(
        tmp_path,
        "train:\n  optimizer:\n    lr: 0.2\n  trainer:\n    epochs: 7\n",
    )
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "lr": 0.9},
        config_path=config_path,
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["lr"] == 0.9
    assert cfg._provenance["lr"] == "flag"
    assert flat["epochs"] == 7
    assert cfg._provenance["epochs"] == "yaml"
    assert flat["weight_decay"] == 0.01
    assert cfg._provenance["weight_decay"] == "default"


def test_flag_and_yaml_merge_within_the_same_group(tmp_path):
    # The crux of the precedence engine: two fields of the SAME group arrive from
    # different layers -- lr from the flag, weight_decay from YAML -- and both must
    # survive the partial-update merge (neither layer's optimizer blob clobbers the
    # other's field).
    config_path = _write(tmp_path, "train:\n  optimizer:\n    weight_decay: 0.5\n")
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "lr": 0.9},
        config_path=config_path,
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["lr"] == 0.9
    assert cfg._provenance["lr"] == "flag"
    assert flat["weight_decay"] == 0.5
    assert cfg._provenance["weight_decay"] == "yaml"


def test_bare_top_level_mapping_loads(tmp_path):
    config_path = _write(tmp_path, "verifier:\n  verifier_name_or_path: from-bare\n")
    cfg = TrainConfig.from_sources(cli={}, config_path=config_path, argv=["train.py"])
    assert cfg.flatten()["verifier_name_or_path"] == "from-bare"
    assert cfg._provenance["verifier_name_or_path"] == "yaml"


@pytest.mark.parametrize("contents", ["", "---\n", "# comment\n"])
def test_empty_or_comment_only_config_is_a_noop(tmp_path, contents):
    # An empty / comment-only --config contributes nothing; the flag still applies.
    config_path = _write(tmp_path, contents)
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m"},
        config_path=config_path,
        argv=["train.py"],
    )
    assert cfg.flatten()["verifier_name_or_path"] == "m"


def test_train_block_wins_over_sibling_stage_keys(tmp_path):
    # A file authored for the future pipeline trains today using only `train:`.
    config_path = _write(
        tmp_path,
        "prepare_data:\n  data_path: ignored\n"
        "launch_vllm:\n  port: 8200\n"
        "train:\n  data:\n    data_path: from-train\n",
    )
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m"}, config_path=config_path, argv=["train.py"]
    )
    assert cfg.flatten()["data_path"] == "from-train"


def test_unknown_key_warns_and_is_ignored(tmp_path):
    config_path = _write(
        tmp_path,
        "train:\n  optimizer:\n    lr: 0.3\n    nonsense_knob: 1\n  bogus_group: {}\n",
    )
    with pytest.warns(UserWarning, match="unrecognised keys"):
        cfg = TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m"},
            config_path=config_path,
            argv=["train.py"],
        )
    assert cfg.flatten()["lr"] == 0.3  # the valid sibling key still applies


def test_yaml_only_draft_init_conflict_is_rejected(tmp_path):
    # The conflict lives entirely in the file: from_pretrained plus a shaping flag.
    config_path = _write(
        tmp_path,
        "train:\n  draft:\n    from_pretrained: ckpt\n    num_layers: 4\n",
    )
    with pytest.raises(ConfigError, match="from-pretrained"):
        TrainConfig.from_sources(cli={}, config_path=config_path, argv=["train.py"])


def test_split_draft_init_conflict_is_rejected(tmp_path):
    # The conflict straddles both layers: the init source in YAML, the shaping flag
    # on the CLI. It must be caught identically to a same-layer conflict, because
    # "explicitly provided" is decided from the layer-agnostic winning-layer record.
    config_path = _write(tmp_path, "train:\n  draft:\n    from_pretrained: ckpt\n")
    with pytest.raises(ConfigError, match="from-pretrained"):
        TrainConfig.from_sources(
            cli={"num_layers": 4}, config_path=config_path, argv=["train.py"]
        )


def test_split_draft_init_conflict_reversed_is_rejected(tmp_path):
    # Same conflict, layers swapped: shaping flag in YAML, init source on the CLI.
    config_path = _write(tmp_path, "train:\n  draft:\n    num_layers: 4\n")
    with pytest.raises(ConfigError, match="from-pretrained"):
        TrainConfig.from_sources(
            cli={"from_pretrained": "ckpt"}, config_path=config_path, argv=["train.py"]
        )


@pytest.mark.parametrize(
    "cli",
    [
        {"from_pretrained": "ckpt"},  # one init source alone
        {"draft_config": "decoder"},  # the other init source alone
        {"num_layers": 4, "draft_arch": "qwen3"},  # shaping flags alone
    ],
)
def test_single_draft_init_source_is_accepted(cli):
    # Exactly one of the three draft-init mechanisms is not a conflict.
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", **cli}, argv=["train.py"]
    )
    assert cfg.speculator_type == "eagle3"


def test_every_flag_is_settable_via_yaml(tmp_path):
    # Each schema dest round-trips through nest_flat -> YAML -> resolved config.
    probes = {
        dest: value
        for dest, value in TrainConfig().flatten().items()
        if isinstance(value, (int, float, str)) and not isinstance(value, bool)
    }
    probes["verifier_name_or_path"] = "m"  # required: give it a non-empty value
    yaml_text = yaml.safe_dump({"train": nest_flat(probes)})
    config_path = _write(tmp_path, yaml_text)
    cfg = TrainConfig.from_sources(cli={}, config_path=config_path, argv=["train.py"])
    for dest in probes:
        assert cfg._provenance[dest] == "yaml", dest


# --------------------------------------------------------------------------- #
# from_sources: algorithm-block mismatch warning
# --------------------------------------------------------------------------- #


def test_mismatched_algorithm_block_warns_via_flag():
    # markov_rank belongs to the dspark group; an eagle3 run ignores it, so warn.
    with pytest.warns(UserWarning, match="does not use the 'dspark'"):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m", "markov_rank": 128}, argv=["train.py"]
        )


def test_mismatched_algorithm_block_warns_via_yaml(tmp_path):
    # The mismatched block is set entirely in YAML: warns identically to the flag.
    config_path = _write(tmp_path, "train:\n  dflash:\n    block_size: 16\n")
    with pytest.warns(UserWarning, match="does not use the 'dflash'"):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m", "speculator_type": "peagle"},
            config_path=config_path,
            argv=["train.py"],
        )


def test_matching_algorithm_block_does_not_warn():
    # dspark consumes both the dflash and dspark groups: setting either is fine.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TrainConfig.from_sources(
            cli={
                "verifier_name_or_path": "m",
                "speculator_type": "dspark",
                "block_size": 16,
                "markov_rank": 128,
            },
            argv=["train.py"],
        )


def test_default_algorithm_block_does_not_warn():
    # An untouched (all-default) mismatched group is not "set", so it stays silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "m", "speculator_type": "eagle3"},
            argv=["train.py"],
        )


# --------------------------------------------------------------------------- #
# resolve: the impure CLI boundary
# --------------------------------------------------------------------------- #


def test_resolve_missing_required_names_the_flag(capsys):
    # No verifier anywhere: the post-build required check surfaces through
    # parser.error as a clean exit(2) whose message still names the flag.
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve([])
    assert exc.value.code == 2
    assert "--verifier-name-or-path" in capsys.readouterr().err


def test_resolve_verifier_from_config_only_succeeds(tmp_path):
    # The reproducibility contract: reloading a run.yaml that supplies the verifier
    # only inside the file must succeed even with no verifier flag on the CLI --
    # the required check runs post-build, after the YAML layer is merged.
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  verifier:\n    verifier_name_or_path: from-yaml\n")
    cfg = TrainConfig.resolve(["--config", str(config)])
    assert cfg.flatten()["verifier_name_or_path"] == "from-yaml"


def test_resolve_config_error_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(
            ["--verifier-name-or-path", "m", "--checkpoint-freq", "1.5"]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid configuration" in err
    assert "Traceback" not in err


def test_resolve_float16_dtype_exits_cleanly(capsys):
    # The dtype validator's float16 rejection surfaces as a clean exit(2).
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(
            ["--verifier-name-or-path", "m", "--hidden-states-dtype", "float16"]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid configuration" in err
    assert "Traceback" not in err


def test_resolve_malformed_config_exits_cleanly(tmp_path, capsys):
    bad = tmp_path / "run.yaml"
    bad.write_text("train:\n  optimizer:\n  lr: [unclosed\n")
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(["--verifier-name-or-path", "m", "--config", str(bad)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert str(bad) in err
    assert "Traceback" not in err


def test_resolve_config_not_a_mapping_exits_cleanly(tmp_path, capsys):
    bad = tmp_path / "run.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(["--verifier-name-or-path", "m", "--config", str(bad)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "top-level mapping" in err
    assert "Traceback" not in err


def test_resolve_flag_overrides_config_file(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  optimizer:\n    lr: 0.2\n")
    cfg = TrainConfig.resolve(
        ["--verifier-name-or-path", "m", "--config", str(config), "--lr", "0.5"]
    )
    assert cfg.flatten()["lr"] == 0.5


def test_help_is_grouped_by_concern():
    titles = {group.title for group in build_parser()._action_groups}
    assert {"general", "verifier", "draft", "data", "optimizer", "mtp"} <= titles


# --------------------------------------------------------------------------- #
# Example-recipe back-compat suite
# --------------------------------------------------------------------------- #


def _config_block(lines: list[str]) -> list[str]:
    """The recipe's ``# ==== Configuration ====`` variable-assignment block."""
    start = next(i for i, line in enumerate(lines) if "Configuration" in line)
    block: list[str] = []
    for line in lines[start + 1 :]:
        if re.match(r"^#\s*=+\s*$", line.strip()):
            break
        block.append(line)
    return block


def _train_invocation_args(lines: list[str]) -> str:
    """The argument text after ``scripts/train.py``, with continuations joined."""
    start = next(i for i, line in enumerate(lines) if "scripts/train.py" in line)
    block: list[str] = []
    for line in lines[start:]:
        block.append(line)
        if not line.rstrip().endswith("\\"):
            break
    joined = " ".join(line.rstrip().rstrip("\\").strip() for line in block)
    return joined.split("scripts/train.py", 1)[1]


def _recipe_argv(path: Path) -> list[str]:
    """Extract the argv a recipe threads into ``train.py`` via a real shell."""
    lines = path.read_text().splitlines()
    script = (
        "\n".join(_config_block(lines))
        + '\n__emit() { printf "%s\\n" "$@"; }\n__emit '
        + _train_invocation_args(lines)
        + "\n"
    )
    out = subprocess.run(  # noqa: S603
        [shutil.which("bash") or "bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


RECIPES: dict[str, dict] = {
    "dflash_qwen3_8b_sharegpt_online_5k.sh": {
        "verifier_name_or_path": "Qwen/Qwen3-8B",
        "data_path": "./output/dflash_qwen3_8b_sharegpt",
        "vllm_endpoint": "http://localhost:8000/v1",
        "save_path": "./output/dflash_qwen3_8b_sharegpt/checkpoints",
        "draft_vocab_size": 32000,
        "epochs": 5,
        "lr": 3e-4,
        "total_seq_len": 8192,
        "speculator_type": "dflash",
        "block_size": 8,
        "max_anchors": 3072,
        "num_layers": 5,
        "target_layer_ids": [2, 18, 33],
        "on_missing": "generate",
        "on_generate": "delete",
    },
    "dspark_qwen3_0_6b_sharegpt_online.sh": {
        "verifier_name_or_path": "Qwen/Qwen3-0.6B",
        "data_path": "./output/dspark_qwen3_0_6b_sharegpt",
        "vllm_endpoint": "http://localhost:8000/v1",
        "save_path": "./output/dspark_qwen3_0_6b_sharegpt/checkpoints",
        "draft_vocab_size": 32000,
        "epochs": 5,
        "lr": 3e-4,
        "total_seq_len": 4096,
        "speculator_type": "dspark",
        "block_size": 8,
        "max_anchors": 3072,
        "num_layers": 3,
        "target_layer_ids": [2, 14, 25],
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "loss_fn": '{"ce": 0.1, "tv": 0.9}',
        "confidence_head_alpha": 1.0,
        "on_missing": "generate",
        "on_generate": "delete",
    },
    "eagle3_llama3_8b_ultrachat_offline_5k.sh": {
        "verifier_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "data_path": "./output",
        "hidden_states_path": "./output/hidden_states",
        "save_path": "./output/checkpoints",
        "draft_vocab_size": 32000,
        "epochs": 5,
        "lr": 1e-4,
        "total_seq_len": 8192,
        "on_missing": "raise",
    },
    "eagle3_qwen3_8b_sharegpt_online_5k.sh": {
        "verifier_name_or_path": "Qwen/Qwen3-8B",
        "data_path": "./output",
        "vllm_endpoint": "http://localhost:8000/v1",
        "save_path": "./output/checkpoints",
        "draft_vocab_size": 32000,
        "epochs": 5,
        "lr": 1e-4,
        "total_seq_len": 8192,
        "on_missing": "generate",
        "on_generate": "delete",
    },
    "mtp_qwen3_5_9b_gsm8k_online.sh": {
        "verifier_name_or_path": "Qwen/Qwen3.5-9B",
        "data_path": "./output",
        "vllm_endpoint": "http://localhost:8000/v1",
        "save_path": "./output/checkpoints",
        "speculator_type": "mtp",
        "num_speculative_steps": 3,
        "target_layer_ids": [32],
        "step_weight_beta": 0.6,
        "epochs": 3,
        "lr": 1e-4,
        "total_seq_len": 8192,
        "on_missing": "generate",
        "on_generate": "delete",
    },
    "peagle_qwen3_8b_sharegpt_online_5k.sh": {
        "verifier_name_or_path": "Qwen/Qwen3-8B",
        "data_path": "./output/peagle_qwen3_8b_sharegpt",
        "vllm_endpoint": "http://localhost:8108/v1",
        "hidden_states_path": "./output/peagle_qwen3_8b_sharegpt/hidden_states",
        "save_path": "./output/peagle_qwen3_8b_sharegpt/checkpoints",
        "epochs": 5,
        "lr": 6e-4,
        "total_seq_len": 4096,
        "speculator_type": "peagle",
        "num_layers": 4,
        "num_depths": 4,
        "down_sample_ratio": 0.7,
        "down_sample_ratio_min": 0.2,
        "norm_before_residual": False,
        "scheduler_type": "cosine",
        "on_missing": "generate",
        "on_generate": "delete",
    },
}


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize(("recipe", "expected"), RECIPES.items(), ids=list(RECIPES))
def test_recipe_flags_resolve_unchanged(recipe: str, expected: dict):
    flat = TrainConfig.resolve(_recipe_argv(EXAMPLES / recipe)).flatten()
    for dest, value in expected.items():
        assert flat[dest] == value, dest
