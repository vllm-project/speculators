# Grafana Dashboard Setup

CI metrics are stored on the `metrics-data` orphan branch as a JSON array of flat `(metric, value, labels)` records. Grafana reads them via the **Infinity** data-source plugin.

## Data model

Every numeric value is its own record. Labels (timestamp, branch, test name, etc.) are auto-populated from the CI environment:

```json
{
  "metric": "acceptance_at_token_0",
  "value": 0.45,
  "timestamp": "2026-11-05T04:12:00+00:00",
  "git_sha": "ab2cab9",
  "branch": "main",
  "build_url": "https://buildkite.com/...",
  "test_id": "test_eagle3_qwen3_8b_sharegpt",
  "test_file": "tests/e2e/regression/test_training_only_acceptance.py"
}
```

This allows any metric name to be recorded without schema changes.

## Prerequisites

1. A Grafana instance (self-hosted or Grafana Cloud free tier).
2. Install the [Infinity data source plugin](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/).

## Data source configuration

| Field  | Value   |
| ------ | ------- |
| Type   | JSON    |
| Source | URL     |
| Parser | Backend |

## Panel queries

Point queries at:

```
https://raw.githubusercontent.com/vllm-project/speculators/metrics-data/metrics.json
```

Columns to extract:

| Selector    | As        | Type      |
| ----------- | --------- | --------- |
| `timestamp` | Time      | Timestamp |
| `metric`    | metric    | String    |
| `value`     | value     | Number    |
| `test_id`   | test_id   | String    |
| `branch`    | branch    | String    |
| `git_sha`   | git_sha   | String    |
| `test_file` | test_file | String    |

Use dashboard variables and **Filter by value** transformations to select specific metrics, tests, or branches.

## One-time setup

Create the orphan branch (requires push access):

```bash
python scripts/ci/push_metrics.py init
```

## Local testing

```bash
echo '[{"metric":"test_metric","value":42,"timestamp":"2026-01-01T00:00:00Z","test_id":"manual"}]' \
  | python scripts/ci/push_metrics.py push - --target metrics.json
```

## Emitting metrics from tests

Set `SPECULATORS_METRICS_DIR` to a directory path. Each test run writes per-test JSON files there, which are then bulk-pushed via `push-dir`:

```bash
export SPECULATORS_METRICS_DIR=/tmp/ci-metrics
pytest tests/e2e/regression/

python scripts/ci/push_metrics.py push-dir "$SPECULATORS_METRICS_DIR" --target metrics.json
```
