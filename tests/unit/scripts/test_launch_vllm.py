from scripts.launch_vllm import (
    DEFAULT_API_SERVER_COUNT,
    DEFAULT_RENDERER_NUM_WORKERS,
    _with_render_defaults,
)


def test_defaults_added_when_absent():
    args = _with_render_defaults(["--port", "8000"])
    assert args == [
        "--api-server-count",
        str(DEFAULT_API_SERVER_COUNT),
        "--renderer-num-workers",
        str(DEFAULT_RENDERER_NUM_WORKERS),
        "--port",
        "8000",
    ]


def test_explicit_flag_follows_default():
    args = _with_render_defaults(["--api-server-count", "1"])
    assert args[-2:] == ["--api-server-count", "1"]


def test_headless_does_not_get_api_server_defaults():
    args = _with_render_defaults(["--headless"])
    assert "--api-server-count" not in args
    assert "--renderer-num-workers" not in args
