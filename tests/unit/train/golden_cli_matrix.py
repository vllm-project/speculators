"""Shared CLI invocation matrix for the proof-of-equivalence golden test.

Each entry is ``(name, argv_tail)`` where ``argv_tail`` are the arguments passed
after ``train.py`` (the required ``--verifier-name-or-path`` is prepended by the
harness). The same matrix is used to (a) capture golden ``vars(args)`` fixtures
from the pre-refactor parser and (b) assert the refactored config layer produces
the identical flat working-dict. See ``test_golden_cli_equivalence.py``.

The matrix intentionally exercises every speculator type and the tricky
draft-init / optimizer / loss / normalization branches -- these are the paths
most likely to drift during the refactor.
"""

# A dummy verifier path keeps ``parse_args()`` fully offline (no model download);
# nothing in argument parsing touches the network.
DUMMY_VERIFIER = "dummy-verifier"

INVOCATIONS: list[tuple[str, list[str]]] = [
    ("eagle3_default", []),
    ("dflash_default", ["--speculator-type", "dflash"]),
    ("dspark_default", ["--speculator-type", "dspark"]),
    ("peagle_default", ["--speculator-type", "peagle"]),
    ("mtp_default", ["--speculator-type", "mtp"]),
    ("from_pretrained", ["--from-pretrained", "/tmp/some/ckpt"]),
    ("draft_config", ["--draft-config", "/tmp/some/decoder"]),
    (
        "dflash_full_attention_indices",
        [
            "--speculator-type",
            "dflash",
            "--sliding-window",
            "1024",
            "--full-attention-indices",
            "0",
            "2",
        ],
    ),
    (
        "scheduler_warmup_ratio",
        [
            "--scheduler-type",
            "cosine",
            "--scheduler-warmup-ratio",
            "0.1",
        ],
    ),
    ("muon_optimizer", ["--optimizer", "muon", "--muon-lr", "0.01"]),
    ("compound_loss", ["--loss-fn", '{"ce": 0.1, "tv": 0.9}']),
    ("dry_run", ["--dry-run"]),
    (
        "norm_toggles",
        ["--no-norm-before-fc", "--no-norm-output", "--fc-norm"],
    ),
    (
        "numeric_overrides",
        [
            "--epochs",
            "5",
            "--lr",
            "2e-4",
            "--num-layers",
            "3",
            "--draft-vocab-size",
            "32000",
            "--total-seq-len",
            "4096",
            "--checkpoint-freq",
            "0.5",
        ],
    ),
    (
        "dspark_heads",
        [
            "--speculator-type",
            "dspark",
            "--markov-rank",
            "128",
            "--markov-head-type",
            "gated",
            "--no-enable-confidence-head",
            "--confidence-head-alpha",
            "0.5",
        ],
    ),
    (
        "peagle_cod",
        [
            "--speculator-type",
            "peagle",
            "--num-depths",
            "4",
            "--down-sample-ratio",
            "0.6",
            "--down-sample-ratio-min",
            "0.1",
        ],
    ),
    # Kitchen sink: toggle every otherwise-untouched bool (store_true and
    # BooleanOptionalAction --no- forms), the list field with real values, and a
    # few non-default Literals -- so the golden proof locks their generated CLI
    # *shape*, not just their defaults.
    (
        "kitchen_sink",
        [
            "--speculator-type",
            "dspark",
            "--save-best",
            "--deterministic-cuda",
            "--no-resume-from-checkpoint",
            "--use-off-policy-tokens",
            "--legacy-data",
            "--trust-remote-code",
            "--embed-requires-grad",
            "--no-norm-before-residual",
            "--no-confidence-head-with-markov",
            "--target-layer-ids",
            "2",
            "3",
            "5",
            "--optimizer",
            "adamw",
            "--scheduler-type",
            "cosine",
            "--markov-head-type",
            "rnn",
        ],
    ),
]


def full_argv(argv_tail: list[str]) -> list[str]:
    """Prepend the program name and required verifier path to an argv tail."""
    return ["train.py", "--verifier-name-or-path", DUMMY_VERIFIER, *argv_tail]
