from scripts.launch_vllm import (
    DEFAULT_RENDERER_NUM_WORKERS,
    _preprocessing_workers,
    _with_render_defaults,
    render_throughput_defaults,
)
from speculators.data_generation.preprocessing import default_preprocessing_workers


def test_defaults_added_when_absent():
    args = _with_render_defaults(["--port", "8000"])
    api_servers, renderer_workers = render_throughput_defaults()
    assert args == [
        "--api-server-count",
        str(api_servers),
        "--renderer-num-workers",
        str(renderer_workers),
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


def test_sizing_preserves_the_384_cpu_optimum():
    assert default_preprocessing_workers(384) == 128
    assert _preprocessing_workers(384) == 128
    assert render_throughput_defaults(384) == (32, DEFAULT_RENDERER_NUM_WORKERS)


def test_sizing_scales_down_on_small_hosts():
    assert default_preprocessing_workers(16) == 8
    assert _preprocessing_workers(16) == 8
    assert render_throughput_defaults(16) == (2, DEFAULT_RENDERER_NUM_WORKERS)
