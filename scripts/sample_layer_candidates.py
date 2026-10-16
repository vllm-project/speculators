"""Sample candidate verifier layer-id sets for hidden-state layer selection.

Draws candidate sets of layer ids without replacement from
``[0, num_layers - 1]`` under a configurable depth bias, enforcing a minimum
gap between the ids within each set. The final layer (id ``num_layers``) is
deliberately outside the pool: ``launch_vllm.py --include-last-layer`` captures
it separately as the verifier's last hidden states.

Each candidate is printed on its own line as space-separated ids, ready to
pass to ``--target-layer-ids``; with more than one bias sampled, each group
is preceded by a ``# <bias>`` comment line. Layer ids index the *input* to
each block: id ``N`` is the output of block ``N - 1`` (id ``0`` is the
embedding output).

Example:

    python scripts/sample_layer_candidates.py \
        --num-layers 32 --num-layer-ids 3 --num-candidates-per-bias 12 \
        --bias deep --minimum-gap 1 --seed 0
"""

import argparse
import random

BIASES = ("uniform", "deep", "shallow", "center", "tails")


def bias_weights(bias: str, num_layers: int) -> list[int]:
    """Per-layer sampling weights over ids ``[0, num_layers - 1]``.

    Zero-weight layers are never sampled. Weights are proportional to:

    - ``uniform``: flat
    - ``deep``: the layer id itself (id 0 is never drawn)
    - ``shallow``: the reverse of ``deep`` (the last id is never drawn)
    - ``center``: a triangle peaking at the middle, zero at both ends
    - ``tails``: the inverse triangle -- zero at the middle, peaks at both ends

    :param bias: One of ``BIASES``.
    :param num_layers: Number of verifier transformer blocks.
    :return: Sampling weight per layer id.
    """
    last = num_layers - 1
    if bias == "uniform":
        return [1] * num_layers
    if bias == "deep":
        return list(range(num_layers))
    if bias == "shallow":
        return [last - i for i in range(num_layers)]
    if bias == "center":
        return [min(i, last - i) for i in range(num_layers)]
    return [abs(2 * i - last) for i in range(num_layers)]


def sample_candidate(
    weights: list[int],
    size: int,
    minimum_gap: int,
    rng: random.Random,
) -> list[int]:
    """Sample one candidate: ``size`` ids with all pairwise gaps satisfied.

    Ids are drawn sequentially from the remaining pool, weighted by
    ``weights`` (as in ``numpy.random.choice(..., replace=False, p=...)``).
    After a pick, it and every id within ``minimum_gap`` of it are removed,
    so any two selected ids ``N`` and ``M`` satisfy ``|N - M| > minimum_gap``.

    :param weights: Per-id sampling weight from :func:`bias_weights`.
    :param size: Number of ids in the candidate (``k``).
    :param minimum_gap: Required pairwise separation, exclusive.
    :param rng: Random generator.
    :return: Sorted list of ``size`` layer ids.
    :raises ValueError: If the pool is exhausted before ``size`` picks.
    """
    allowed = [True] * len(weights)
    picked: list[int] = []
    for _ in range(size):
        total = sum(w for w, ok in zip(weights, allowed, strict=True) if ok)
        if total <= 0:
            raise ValueError(
                "no eligible layers left to sample; reduce"
                " --num-layer-ids or --minimum-gap, or pick a --bias"
                " with more positive-weight layers at this depth"
            )
        threshold = rng.random() * total
        cumulative = 0
        for i, (w, ok) in enumerate(zip(weights, allowed, strict=True)):
            if not ok:
                continue
            cumulative += w
            if threshold < cumulative:
                picked.append(i)
                lo = max(0, i - minimum_gap)
                hi = min(len(weights), i + minimum_gap + 1)
                allowed[lo:hi] = [False] * (hi - lo)
                break
    return sorted(picked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample candidate verifier hidden-state layer-id sets for"
            " layer-selection ablations. Each line of output is one candidate,"
            " as space-separated ids usable with --target-layer-ids, grouped"
            " under # <bias> comment headers when several biases are sampled."
        )
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
        help="Number of candidate sets (M) sampled for each bias.",
    )
    parser.add_argument(
        "--bias",
        nargs="+",
        choices=BIASES,
        default=list(BIASES),
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
    args = parser.parse_args()

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

    biases = list(dict.fromkeys(args.bias))  # dedupe, keep order
    per_bias_weights: dict[str, list[int]] = {}
    for bias in biases:
        weights = bias_weights(bias, args.num_layers)
        if sum(weights) <= 0:
            parser.error(
                f"--bias {bias} leaves every layer with zero weight at"
                f" --num-layers {args.num_layers}"
            )
        per_bias_weights[bias] = weights

    rng = random.Random(args.seed)
    try:
        for bias, weights in per_bias_weights.items():
            print(f"# {bias}")
            for _ in range(args.num_candidates_per_bias):
                candidate = sample_candidate(
                    weights, args.num_layer_ids, args.minimum_gap, rng
                )
                print(" ".join(map(str, candidate)))
    except ValueError as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
