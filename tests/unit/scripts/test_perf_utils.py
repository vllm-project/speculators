"""Tests for scripts/evaluate/perf_utils.py.

Covers the changed code paths from the guidellm 0.6→0.7 upgrade:
  - parse_gen_kwargs (replaced build_backend_args)
  - run_guidellm CLI command construction
  - _load_json (new JSON output structure)
  - parse_gen_len_file (new request stats structure)
  - parse_sweep_file (unchanged, regression guard)
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "evaluate"
_PERF_UTILS_PATH = _SCRIPT_DIR / "perf_utils.py"


@pytest.fixture(scope="module")
def perf_utils():
    spec = importlib.util.spec_from_file_location(
        "perf_utils", _PERF_UTILS_PATH, submodule_search_locations=[]
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    prev = sys.modules.get("perf_utils")
    sys.modules["perf_utils"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if prev is None:
            sys.modules.pop("perf_utils", None)
        else:
            sys.modules["perf_utils"] = prev
        raise
    return module


# ---------------------------------------------------------------------------
# parse_gen_kwargs
# ---------------------------------------------------------------------------


class TestParseGenKwargs:
    def test_empty_string(self, perf_utils):
        assert perf_utils.parse_gen_kwargs("") == {}

    def test_valid_json(self, perf_utils):
        result = perf_utils.parse_gen_kwargs('{"temperature": 0.6, "top_p": 0.9}')
        assert result == {"temperature": 0.6, "top_p": 0.9}

    def test_invalid_json_raises(self, perf_utils):
        with pytest.raises(ValueError, match="Invalid JSON"):
            perf_utils.parse_gen_kwargs("{bad json}")


# ---------------------------------------------------------------------------
# run_guidellm — command construction
# ---------------------------------------------------------------------------


class TestRunGuidellm:
    def _capture_cmd(self, perf_utils, **kwargs):
        defaults = {
            "target": "http://localhost:8000/v1",
            "dataset": "RedHatAI/speculator_benchmarks",
            "subset": "qa",
            "data_column_mapper": (
                "kind=generative_column_mapper,column_mappings.text_column=prompt"
            ),
            "profile": "sweep",
            "rate": 10,
            "max_requests": 200,
            "max_concurrency": 128,
            "output_path": Path("/tmp/out.json"),
            "max_tokens": 4096,
            "gen_kwargs": None,
        }
        defaults.update(kwargs)
        with patch("subprocess.run") as mock_run:
            perf_utils.run_guidellm(**defaults)
            return mock_run.call_args[0][0]

    def test_subcommand_is_run(self, perf_utils):
        cmd = self._capture_cmd(perf_utils)
        assert cmd[0] == "guidellm"
        assert cmd[1] == "run"

    def test_backend_flag(self, perf_utils):
        cmd = self._capture_cmd(perf_utils)
        idx = cmd.index("--backend")
        backend = cmd[idx + 1]
        assert "kind=openai_http" in backend
        assert "target=http://localhost:8000/v1" in backend
        assert "max_tokens=4096" in backend

    def test_backend_gen_kwargs(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, gen_kwargs={"temperature": 0.6})
        idx = cmd.index("--backend")
        backend = cmd[idx + 1]
        assert "extras.body.temperature=0.6" in backend

    def test_data_huggingface_with_subset(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, subset="qa")
        idx = cmd.index("--data")
        data = cmd[idx + 1]
        assert "kind=huggingface" in data
        assert "source=RedHatAI/speculator_benchmarks" in data
        assert "load_kwargs.data_files=qa.jsonl" in data

    def test_data_local_file_without_subset(self, perf_utils):
        cmd = self._capture_cmd(
            perf_utils,
            subset=None,
            dataset="/tmp/local.jsonl",
        )
        idx = cmd.index("--data")
        data = cmd[idx + 1]
        assert "kind=json_file" in data
        assert "path=/tmp/local.jsonl" in data

    def test_profile_sweep(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, profile="sweep", rate=10)
        idx = cmd.index("--profile")
        profile = cmd[idx + 1]
        assert "kind=sweep" in profile
        assert "sweep_size=10" in profile
        assert "max_concurrency=128" in profile

    def test_profile_throughput_no_sweep_size(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, profile="throughput", rate=128)
        idx = cmd.index("--profile")
        profile = cmd[idx + 1]
        assert "kind=throughput" in profile
        assert "sweep_size" not in profile

    def test_constraint_max_requests(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, max_requests=200)
        idx = cmd.index("--constraint")
        constraint = cmd[idx + 1]
        assert "kind=max_requests" in constraint
        assert "count=200" in constraint

    def test_no_constraint_when_max_requests_none(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, max_requests=None)
        assert "--constraint" not in cmd

    def test_output_flag(self, perf_utils):
        cmd = self._capture_cmd(perf_utils, output_path=Path("/tmp/out.json"))
        idx = cmd.index("--output")
        output = cmd[idx + 1]
        assert "kind=json" in output
        assert "path=/tmp/out.json" in output


# ---------------------------------------------------------------------------
# _load_json — JSON output parsing
# ---------------------------------------------------------------------------


def _make_benchmark_json(
    subset_file="qa.jsonl",
    strategy_type="constant",
    rps_mean=50.0,
    latency_median=0.1,
):
    return {
        "config": {
            "spec": {
                "data": [
                    {
                        "kind": "huggingface",
                        "source": "RedHatAI/speculator_benchmarks",
                        "load_kwargs": {"data_files": subset_file},
                    }
                ]
            }
        },
        "benchmarks": [
            {
                "config": {
                    "strategy": {"type_": strategy_type, "rate": 50.0},
                },
                "metrics": {
                    "requests_per_second": {
                        "successful": {"mean": rps_mean},
                    },
                    "request_latency": {
                        "successful": {"median": latency_median},
                    },
                    "inter_token_latency_ms": {
                        "successful": {"median": 5.0},
                    },
                    "time_to_first_token_ms": {
                        "successful": {"median": 20.0},
                    },
                    "output_tokens_per_second": {
                        "successful": {"median": 100.0},
                    },
                },
            }
        ],
    }


class TestLoadJson:
    def test_extracts_subset_from_data_config(self, perf_utils, tmp_path):
        data = _make_benchmark_json(subset_file="HumanEval.jsonl")
        fp = tmp_path / "bench.json"
        fp.write_text(json.dumps(data))
        result = perf_utils._load_json(fp, "latency")
        assert "HumanEval" in result

    def test_extracts_latency_points(self, perf_utils, tmp_path):
        data = _make_benchmark_json(rps_mean=50.0, latency_median=0.1)
        fp = tmp_path / "bench.json"
        fp.write_text(json.dumps(data))
        result = perf_utils._load_json(fp, "latency")
        assert result["qa"] == [(50.0, 0.1)]

    def test_skips_non_constant_strategies(self, perf_utils, tmp_path):
        data = _make_benchmark_json(strategy_type="throughput")
        fp = tmp_path / "bench.json"
        fp.write_text(json.dumps(data))
        result = perf_utils._load_json(fp, "latency")
        assert result == {}

    def test_multiple_benchmarks_sorted(self, perf_utils, tmp_path):
        data = _make_benchmark_json()
        data["benchmarks"].append(
            {
                "config": {"strategy": {"type_": "constant", "rate": 100.0}},
                "metrics": {
                    "requests_per_second": {"successful": {"mean": 20.0}},
                    "request_latency": {"successful": {"median": 0.2}},
                },
            }
        )
        fp = tmp_path / "bench.json"
        fp.write_text(json.dumps(data))
        result = perf_utils._load_json(fp, "latency")
        points = result["qa"]
        assert points == [(20.0, 0.2), (50.0, 0.1)]


# ---------------------------------------------------------------------------
# parse_gen_len_file — request stats parsing
# ---------------------------------------------------------------------------


def _make_gen_len_json(output_token_counts):
    return {
        "benchmarks": [
            {
                "requests": {
                    "successful": [
                        {"output_metrics": {"text_tokens": n}}
                        for n in output_token_counts
                    ]
                }
            }
        ]
    }


class TestParseGenLenFile:
    def test_basic_stats(self, perf_utils, tmp_path):
        fp = tmp_path / "gen_len.json"
        fp.write_text(json.dumps(_make_gen_len_json([100, 200, 300])))
        result = perf_utils.parse_gen_len_file(fp)
        assert result["count"] == 3
        assert result["median"] == 200
        assert result["min"] == 100
        assert result["max"] == 300

    def test_max_tokens_power_of_two(self, perf_utils, tmp_path):
        fp = tmp_path / "gen_len.json"
        fp.write_text(json.dumps(_make_gen_len_json([100, 200, 300])))
        result = perf_utils.parse_gen_len_file(fp)
        assert result["max_tokens"] == 256  # 2^ceil(log2(200))

    def test_no_benchmarks_raises(self, perf_utils, tmp_path):
        fp = tmp_path / "gen_len.json"
        fp.write_text(json.dumps({"benchmarks": []}))
        with pytest.raises(ValueError, match="No benchmarks"):
            perf_utils.parse_gen_len_file(fp)

    def test_no_successful_requests_raises(self, perf_utils, tmp_path):
        fp = tmp_path / "gen_len.json"
        fp.write_text(json.dumps({"benchmarks": [{"requests": {"successful": []}}]}))
        with pytest.raises(ValueError, match="No successful requests"):
            perf_utils.parse_gen_len_file(fp)


# ---------------------------------------------------------------------------
# parse_sweep_file — regression guard (unchanged logic)
# ---------------------------------------------------------------------------


def _make_sweep_json(subset_name="qa"):
    return {
        "benchmarks": [
            {
                "config": {
                    "strategy": {"type_": "constant", "rate": 10.0},
                },
                "metrics": {
                    "requests_per_second": {
                        "successful": {"median": 9.5},
                    },
                    "request_latency": {
                        "successful": {"median": 0.15},
                    },
                    "inter_token_latency_ms": {
                        "successful": {"median": 4.2},
                    },
                    "time_to_first_token_ms": {
                        "successful": {"median": 18.0},
                    },
                    "output_tokens_per_second": {
                        "successful": {"median": 95.0},
                    },
                    "output_tokens": {
                        "successful": {"sum": 50000},
                    },
                },
            },
            {
                "config": {
                    "strategy": {"type_": "throughput"},
                },
                "metrics": {},
            },
        ],
    }


class TestParseSweepFile:
    def test_extracts_constant_rows(self, perf_utils, tmp_path):
        fp = tmp_path / "sweep_qa.json"
        fp.write_text(json.dumps(_make_sweep_json()))
        rows = perf_utils.parse_sweep_file(fp)
        assert len(rows) == 1
        assert rows[0]["strategy"] == "constant"
        assert rows[0]["target_rate"] == 10.0

    def test_skips_throughput_strategy(self, perf_utils, tmp_path):
        fp = tmp_path / "sweep_qa.json"
        fp.write_text(json.dumps(_make_sweep_json()))
        rows = perf_utils.parse_sweep_file(fp)
        strategies = [r["strategy"] for r in rows]
        assert "throughput" not in strategies

    def test_subset_from_filename(self, perf_utils, tmp_path):
        fp = tmp_path / "sweep_HumanEval.json"
        fp.write_text(json.dumps(_make_sweep_json()))
        rows = perf_utils.parse_sweep_file(fp)
        assert rows[0]["subset"] == "HumanEval"
