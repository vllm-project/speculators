from collections.abc import Sequence


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(len(ordered) * fraction + 0.999999) - 1),
    )
    return float(ordered[index])
