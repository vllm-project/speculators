"""Re-type a converted DFlash checkpoint as an XPress starting point.

``convert_model()`` knows nothing about XPress, so a warm start is two steps:
convert the DFlash backbone normally, then run this to switch the config over and
stamp the fields that live in the MODEL CONFIG rather than on the CLI. Keeping it
separate leaves the upstream converter untouched.

    python examples/train/xpress_morph_config.py <converted_dir> [--shift]

Idempotent, and meant to be re-run: a checkpoint morphed by an older revision
would otherwise silently miss any field added since.

``--rank`` / ``--mlp-ratio`` size the refiner head. They matter here and nowhere
else: a ``--from-pretrained`` run reads its architecture from this config, so the
CLI flags of the same name are ignored on that path. The defaults (256 / 2, an MLP
hidden of 512) are what the released checkpoints use.

``--shift`` selects the DeepSeek/DSpark block convention
(``sample_from_anchor=True``, every slot predicts the next token). The default is
fill-in, which is what z-lab checkpoints are native to: slot 0 carries the known
anchor and slots 1..B-1 predict.
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="converted checkpoint dir")
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    args = parser.parse_args()

    path = args.checkpoint / "config.json"
    cfg = json.loads(path.read_text())
    model_type = cfg.get("speculators_model_type")
    if model_type not in ("dflash", "xpress"):
        raise SystemExit(
            f"expected a converted dflash/xpress checkpoint, got {model_type!r}"
        )

    cfg["speculators_model_type"] = "xpress"
    cfg["architectures"] = ["XPressDraftModel"]
    cfg["sample_from_anchor"] = args.shift
    cfg["xpress_rank"] = args.rank
    cfg["xpress_mlp_ratio"] = args.mlp_ratio
    cfg.setdefault("num_jacobi_passes", 6)
    cfg.setdefault("mask_token_id", 151669)
    path.write_text(json.dumps(cfg, indent=2))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(
        "morphed %s: xpress, %s, block_size=%s, rank=%d, mlp_hidden=%d",
        path,
        "shift" if args.shift else "fill-in",
        cfg.get("block_size"),
        args.rank,
        args.rank * args.mlp_ratio,
    )


if __name__ == "__main__":
    main()
