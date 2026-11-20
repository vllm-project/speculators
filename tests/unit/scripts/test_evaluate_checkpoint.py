"""Serving adapter boundaries that do not require vLLM or AIPerf."""

import importlib.util
import json
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "examples/train/evaluate_checkpoint.py"


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location("evaluate_checkpoint", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserves_each_datasets_pooled_acceptance_values(adapter, tmp_path):
    csv = tmp_path / "acceptance.csv"
    csv.write_text(
        "dataset,acceptance_length,acceptance_rate_pct,acceptance_rate_pos_1_pct\n"
        "gsm8k,3.25,75,90\n"
        "humaneval,4.5,80,95\n"
    )
    assert adapter.read_acceptance(csv) == {
        "gsm8k/acceptance_length": 3.25,
        "gsm8k/acceptance_rate_pct": 75,
        "gsm8k/acceptance_rate_pos_1_pct": 90,
        "humaneval/acceptance_length": 4.5,
        "humaneval/acceptance_rate_pct": 80,
        "humaneval/acceptance_rate_pos_1_pct": 95,
    }


@pytest.mark.parametrize(
    "rows",
    [
        "dataset,acceptance_length,acceptance_rate_pct\n",
        "dataset,acceptance_length,acceptance_rate_pct\ngsm8k,,\n",
        "dataset,acceptance_length,acceptance_rate_pct\ngsm8k,nan,75\n",
        "dataset,acceptance_length,acceptance_rate_pct\ngsm8k,3,75\ngsm8k,4,80\n",
    ],
)
def test_rejects_unusable_or_ambiguous_acceptance_reports(adapter, tmp_path, rows):
    csv = tmp_path / "acceptance.csv"
    csv.write_text(rows)
    with pytest.raises(ValueError):
        adapter.read_acceptance(csv)


@pytest.mark.parametrize("model", ["expected-draft", "unrelated-draft"])
def test_readiness_checks_the_launched_model_identity(adapter, model):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": model}]}).encode())

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        if model == "expected-draft":
            adapter.wait_ready(process, url, timeout=2, expected_model="expected-draft")
        else:
            with pytest.raises(RuntimeError, match="identity"):
                adapter.wait_ready(
                    process, url, timeout=2, expected_model="expected-draft"
                )
    finally:
        process.terminate()
        process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_does_not_use_an_existing_server_after_launch_failed(adapter):
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    process.wait(timeout=5)
    with pytest.raises(RuntimeError, match="exited"):
        adapter.wait_ready(
            process, "http://127.0.0.1:1", timeout=2, expected_model="expected-draft"
        )


def test_help_documents_required_separate_gpu_selection():
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--cuda-visible-devices" in result.stdout


def test_adapter_passes_checkpoint_algorithm_and_unique_model_to_existing_tools(
    adapter, tmp_path, monkeypatch
):
    scripts = tmp_path / "scripts"
    scripts.joinpath("evaluate").mkdir(parents=True)
    scripts.joinpath("launch_vllm.py").write_text(
        textwrap.dedent("""\
        import json, os, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path
        args = sys.argv[1:]
        assert args[args.index("--spec-method") + 1] == "eagle3"
        checkpoint = os.environ["SPECULATORS_EVAL_CHECKPOINT"]
        assert args[args.index("--spec-model") + 1] == checkpoint
        model = args[args.index("--served-model-name") + 1]
        port = int(args[args.index("--port") + 1])
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": model}]}).encode())
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """)
    )
    scripts.joinpath("evaluate/evaluate.py").write_text(
        textwrap.dedent("""\
        import json, sys
        from pathlib import Path
        args = sys.argv[1:]
        model = args[args.index("--model") + 1]
        assert model.startswith("speculators-eval-")
        assert args[args.index("--tokenizer") + 1] == "target"
        out = Path(args[args.index("--output-artifact-dir") + 1])
        out.mkdir()
        out.joinpath("acceptance.csv").write_text("dataset,acceptance_length\\ngsm8k,3.25\\n")
    """)
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_text(
        '{"speculators_config": {"algorithm": "eagle3"}}'
    )
    output = tmp_path / "output"
    output.mkdir()
    port_server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = port_server.server_port
    port_server.server_close()
    monkeypatch.setenv("SPECULATORS_EVAL_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SPECULATORS_EVAL_OUTPUT_DIR", str(output))
    monkeypatch.setattr(adapter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--verifier",
            "target",
            "--cuda-visible-devices",
            "1",
            "--port",
            str(port),
            "--startup-timeout",
            "3",
        ],
    )

    adapter.main()

    assert json.loads(output.joinpath("metrics.json").read_text()) == {
        "gsm8k/acceptance_length": 3.25
    }
