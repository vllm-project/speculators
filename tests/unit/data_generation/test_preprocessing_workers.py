import os

from speculators.data_generation.preprocessing import (
    CPUS_PER_PREPROCESSING_WORKER,
    MAX_PREPROCESSING_WORKERS,
    MIN_PREPROCESSING_WORKERS,
    default_preprocessing_workers,
    usable_cpu_count,
)


def test_small_hosts_keep_the_previous_fixed_default():
    # Rendering is client-bound, so a small host should not render with fewer
    # workers than it did before this became CPU-derived.
    for cpus in (1, 2, 4, 8, 16):
        assert default_preprocessing_workers(cpus) == MIN_PREPROCESSING_WORKERS


def test_large_hosts_scale_with_cpus():
    assert default_preprocessing_workers(384) == 384 // CPUS_PER_PREPROCESSING_WORKER
    assert default_preprocessing_workers(96) == 96 // CPUS_PER_PREPROCESSING_WORKER


def test_capped_at_the_measured_knee():
    # 256 workers measured slower than 128 on a 384-CPU node.
    assert default_preprocessing_workers(100_000) == MAX_PREPROCESSING_WORKERS


def test_never_decreases_with_more_cpus():
    counts = [default_preprocessing_workers(c) for c in range(1, 1024)]
    assert counts == sorted(counts)


def test_usable_cpu_count_is_positive_and_bounded():
    cpus = usable_cpu_count()
    assert cpus >= 1
    # Never more than the machine has: the point is to respect affinity, so it
    # must not over-report when the process is pinned.
    assert cpus <= (os.cpu_count() or 1)
