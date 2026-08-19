from scripts.launch_vllm import (
    CPUS_PER_API_SERVER,
    RENDER_CLIENT_CONCURRENCY,
    RENDERER_THREADS_PER_SERVER,
    _with_render_defaults,
    render_throughput_defaults,
)


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


def test_small_hosts_get_fewer_front_ends():
    # One front end per CPUS_PER_API_SERVER cores: on a small box the extra
    # processes only take cores away from the preprocessing workers.
    assert (
        render_throughput_defaults(CPUS_PER_API_SERVER * 2)["--api-server-count"] == "2"
    )
    assert render_throughput_defaults(CPUS_PER_API_SERVER)["--api-server-count"] == "1"


def test_never_fewer_than_one_front_end():
    for cpus in (0, 1, 2):
        assert render_throughput_defaults(cpus)["--api-server-count"] == "1"


def test_large_hosts_stop_adding_front_ends():
    # Only RENDER_CLIENT_CONCURRENCY renders are ever in flight, so past the
    # point where the front end can serve them a bigger host gets the same
    # config -- more API processes would be ~1 GB of RSS each for nothing.
    big = render_throughput_defaults(4096)
    assert big == render_throughput_defaults(64)
    slots = int(big["--api-server-count"]) * RENDERER_THREADS_PER_SERVER
    assert slots <= RENDER_CLIENT_CONCURRENCY


def test_defaults_never_decrease_with_more_cpus():
    counts = [
        int(render_throughput_defaults(c)["--api-server-count"]) for c in range(1, 512)
    ]
    assert counts == sorted(counts)


def test_renderer_threads_always_set():
    assert render_throughput_defaults(1)["--renderer-num-workers"] == str(
        RENDERER_THREADS_PER_SERVER
    )
