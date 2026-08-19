from scripts.launch_vllm import RENDER_THROUGHPUT_DEFAULTS, _with_render_defaults


def test_defaults_added_when_absent():
    args = _with_render_defaults(["--port", "8000"])
    for flag, value in RENDER_THROUGHPUT_DEFAULTS.items():
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
