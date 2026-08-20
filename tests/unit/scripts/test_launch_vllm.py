from scripts.launch_vllm import (
    CPUS_PER_API_SERVER,
    MAX_API_SERVERS,
    RENDERER_THREADS_PER_SERVER,
    WORKERS_PER_API_SERVER,
    _preprocessing_workers,
    _with_render_defaults,
    render_throughput_defaults,
)
from speculators.data_generation.preprocessing import default_preprocessing_workers


def test_defaults_added_when_absent():
    args = _with_render_defaults(["--port", "8000"])
    for flag, value in render_throughput_defaults().items():
        assert args[args.index(flag) + 1] == value
    # Defaults precede user args so an explicit flag wins on duplicates.
    assert args[-2:] == ["--port", "8000"]


def test_explicit_flag_suppresses_default():
    args = _with_render_defaults(["--api-server-count", "1"])
    assert args.count("--api-server-count") == 1
    assert "--renderer-num-workers" in args  # the unset flag still defaults


def test_underscore_and_equals_spellings_suppress_default():
    args = _with_render_defaults(
        ["--api_server_count=1", "--renderer_num_workers", "1"]
    )
    assert "--api-server-count" not in args
    assert "--renderer-num-workers" not in args


def test_worker_formula_matches_the_preprocessing_side():
    # launch_vllm.py runs in the vLLM virtualenv and cannot import speculators
    # (#958, #1008), so it carries its own copy of this formula. If the two ever
    # drift the front end is sized for a client concurrency that never happens.
    for cpus in (1, 2, 4, 8, 16, 32, 64, 128, 192, 384, 1024):
        assert _preprocessing_workers(cpus) == default_preprocessing_workers(cpus)


def test_front_end_tracks_client_concurrency():
    # One API server per WORKERS_PER_API_SERVER renders in flight, so the front
    # end grows with the clients rather than with the machine.
    for cpus in (64, 128, 192, 384):
        api = int(render_throughput_defaults(cpus)["--api-server-count"])
        expected = _preprocessing_workers(cpus) // WORKERS_PER_API_SERVER
        assert api == min(expected, MAX_API_SERVERS, cpus // CPUS_PER_API_SERVER)


def test_small_hosts_get_fewer_front_ends():
    # On a small box extra front ends only take cores from the workers.
    assert render_throughput_defaults(4)["--api-server-count"] == "1"
    assert render_throughput_defaults(16)["--api-server-count"] == "2"


def test_never_fewer_than_one_front_end():
    for cpus in (0, 1, 2):
        assert render_throughput_defaults(cpus)["--api-server-count"] == "1"


def test_front_end_count_is_capped():
    assert render_throughput_defaults(1_000_000)["--api-server-count"] == str(
        MAX_API_SERVERS
    )


def test_defaults_never_decrease_with_more_cpus():
    counts = [
        int(render_throughput_defaults(c)["--api-server-count"]) for c in range(1, 512)
    ]
    assert counts == sorted(counts)


def test_renderer_threads_always_set():
    assert render_throughput_defaults(1)["--renderer-num-workers"] == str(
        RENDERER_THREADS_PER_SERVER
    )
