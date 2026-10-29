"""Sample candidate verifier layer-id sets for hidden-state layer selection.

Draws candidate sets of layer ids without replacement from
``[0, num_layers - 1]`` under a configurable depth bias, enforcing a minimum
gap between the ids within each set. The final layer (id ``num_layers``) is
deliberately outside the pool: ``launch_vllm.py --include-last-layer`` captures
it separately as the verifier's last hidden states.

Each candidate is printed on its own line as space-separated ids, ready to
pass to ``--target-layer-ids``; with more than one bias type sampled, each
group is preceded by a ``# <bias-type>`` comment line. Layer ids index the
*input* to each block: id ``N`` is the output of block ``N - 1`` (id ``0`` is
the embedding output).

Example:

    python scripts/sample_layer_candidates.py \
        --num-layers 32 --num-layer-ids 3 --num-candidates-per-bias 12 \
        --bias-type deep --minimum-gap 1 --seed 0
"""

import argparse
import random

BIAS_TYPES = ("uniform", "deep", "shallow", "center", "tails")


def bias_type_weights(bias_type: str, num_layers: int) -> list[int]:
    """Per-layer sampling weights over ids ``[0, num_layers - 1]``.

    Zero-weight layers are never sampled. Weights are proportional to:

    - ``uniform``: flat
    - ``deep``: the layer id itself (id 0 is never drawn)
    - ``shallow``: the reverse of ``deep`` (the last id is never drawn)
    - ``center``: a triangle peaking at the middle, zero at both ends
    - ``tails``: the inverse triangle -- zero at the middle, peaks at both ends

    :param bias_type: One of ``BIAS_TYPES``.
    :param num_layers: Number of verifier transformer blocks.
    :return: Sampling weight per layer id.
    """
    last = num_layers - 1
    if bias_type == "uniform":
        return [1] * num_layers
    if bias_type == "deep":
        return list(range(num_layers))
    if bias_type == "shallow":
        return [last - i for i in range(num_layers)]
    if bias_type == "center":
        return [min(i, last - i) for i in range(num_layers)]
    return [abs(2 * i - last) for i in range(num_layers)]


def _max_picks(ids: list[int], minimum_gap: int) -> int:
    """Largest number of ``ids`` pairwise separated by more than the gap.

    ``ids`` must be sorted; the greedy left-to-right scan is optimal.
    """
    count = 0
    prev: int | None = None
    for i in ids:
        if prev is None or i - prev > minimum_gap:
            count += 1
            prev = i
    return count


def sample_candidate(
    weights: list[int],
    size: int,
    minimum_gap: int,
    rng: random.Random,
) -> list[int]:
    """Sample one candidate: ``size`` ids with all pairwise gaps satisfied.

    Ids are drawn sequentially from the remaining pool, weighted by
    ``weights`` (as in ``numpy.random.choice(..., replace=False, p=...)``).
    Only ids that keep the candidate completable -- whose gap exclusion
    still leaves enough mutually separated eligible ids for the remaining
    picks -- are considered, so a feasible request never dead-ends. After a
    pick, it and every id within ``minimum_gap`` of it are removed, so any
    two selected ids ``N`` and ``M`` satisfy ``|N - M| > minimum_gap``.

    :param weights: Per-id sampling weight from :func:`bias_type_weights`.
    :param size: Number of ids in the candidate (``k``).
    :param minimum_gap: Required pairwise separation, exclusive.
    :param rng: Random generator.
    :return: Sorted list of ``size`` layer ids.
    :raises ValueError: If no valid candidate of ``size`` ids exists.
    """
    allowed = [True] * len(weights)
    picked: list[int] = []
    for _ in range(size):
        remaining = size - len(picked) - 1
        # Ids that can still be picked: allowed with positive weight.
        eligible = [
            i
            for i, (w, ok) in enumerate(zip(weights, allowed, strict=True))
            if ok and w > 0
        ]
        # A pick is kept only if its gap exclusion still leaves at least
        # ``remaining`` mutually separated eligible ids; otherwise the
        # loop could dead-end even though a valid candidate exists.
        safe = [
            i
            for i in eligible
            if _max_picks(
                [j for j in eligible if not (i - minimum_gap <= j <= i + minimum_gap)],
                minimum_gap,
            )
            >= remaining
        ]
        if not safe:
            raise ValueError(
                "no eligible layers left to sample; reduce"
                " --num-layer-ids or --minimum-gap, or pick a --bias-type"
                " with more positive-weight layers at this depth"
            )
        total = sum(weights[i] for i in safe)
        threshold = rng.random() * total
        cumulative = 0
        for i in safe:
            cumulative += weights[i]
            if threshold < cumulative:
                picked.append(i)
                lo = max(0, i - minimum_gap)
                hi = min(len(weights), i + minimum_gap + 1)
                allowed[lo:hi] = [False] * (hi - lo)
                break
    return sorted(picked)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample candidate verifier hidden-state layer-id sets for"
            " layer-selection ablations. Each line of output is one candidate,"
            " as space-separated ids usable with --target-layer-ids, grouped"
            " under # <bias-type> comment headers when several bias types"
            " are sampled."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help=(
            "Number of verifier transformer blocks; ids span"
            " [0, num_layers - 1]. The final layer is not in the pool:"
            " --include-last-layer captures it separately."
        ),
    )
    parser.add_argument(
        "--num-layer-ids",
        type=int,
        required=True,
        help="Number of layer ids (k) in each candidate set.",
    )
    parser.add_argument(
        "--num-candidates-per-bias",
        type=int,
        required=True,
        help="Number of candidate sets (M) sampled for each bias type.",
    )
    parser.add_argument(
        "--bias-type",
        nargs="+",
        choices=BIAS_TYPES,
        default=list(BIAS_TYPES),
        help=(
            "One or more depth weightings of the sampling distribution"
            " (default: all). deep favors late layers (P proportional to"
            " id), shallow favors early ones, center peaks mid-depth with"
            " zero weight at both ends, tails is the inverse of center."
        ),
    )
    parser.add_argument(
        "--minimum-gap",
        type=int,
        default=1,
        help=(
            "Enforce |N - M| > minimum-gap for any two ids N, M in a"
            " candidate (default: 1, which excludes adjacent layers;"
            " pass 0 to allow them)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible candidates.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid argument values via ``parser.error``."""
    if args.num_layers < 1:
        parser.error("--num-layers must be >= 1")
    if args.num_layer_ids < 1:
        parser.error("--num-layer-ids must be >= 1")
    if args.num_candidates_per_bias < 1:
        parser.error("--num-candidates-per-bias must be >= 1")
    if args.minimum_gap < 0:
        parser.error("--minimum-gap must be >= 0")
    # At most ceil(num_layers / (minimum_gap + 1)) ids can coexist under the
    # pairwise gap constraint.
    max_size = -(-args.num_layers // (args.minimum_gap + 1))
    if args.num_layer_ids > max_size:
        parser.error(
            f"--num-layer-ids {args.num_layer_ids} is infeasible:"
            f" at most {max_size} ids from {args.num_layers} layers can satisfy"
            f" |N - M| > {args.minimum_gap}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    bias_types = list(dict.fromkeys(args.bias_type))  # dedupe, keep order
    per_bias_type_weights: dict[str, list[int]] = {}
    for bias_type in bias_types:
        weights = bias_type_weights(bias_type, args.num_layers)
        if sum(weights) <= 0:
            parser.error(
                f"--bias-type {bias_type} leaves every layer with zero weight"
                f" at --num-layers {args.num_layers}"
            )
        per_bias_type_weights[bias_type] = weights

    rng = random.Random(args.seed)
    try:
        for bias_type, weights in per_bias_type_weights.items():
            # Header only when several bias types are sampled, so output
            # for a single bias type is purely numeric (parseable by
            # --target-layer-ids).
            if len(per_bias_type_weights) > 1:
                print(f"# {bias_type}")
            for _ in range(args.num_candidates_per_bias):
                candidate = sample_candidate(
                    weights, args.num_layer_ids, args.minimum_gap, rng
                )
                print(" ".join(map(str, candidate)))
    except ValueError as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
