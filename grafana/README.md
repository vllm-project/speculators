# Grafana Dashboard Setup

CI regression tests push acceptance metrics to the `metrics-data` orphan branch as a JSON array. Grafana reads them via the **Infinity** data-source plugin.

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

### Acceptance rate over time

| Setting | Value                                                                                     |
| ------- | ----------------------------------------------------------------------------------------- |
| URL     | `https://raw.githubusercontent.com/vllm-project/speculators/metrics-data/acceptance.json` |
| Type    | JSON                                                                                      |
| Parser  | Backend                                                                                   |
| Format  | Dataframe                                                                                 |

Columns to extract (add via **Columns** in the query editor):

| Selector                | As         | Type      |
| ----------------------- | ---------- | --------- |
| `timestamp`             | Time       | Timestamp |
| `test_id`               | test_id    | String    |
| `acceptance_at_token_0` | pos_0      | Number    |
| `acceptance_at_token_1` | pos_1      | Number    |
| `acceptance_at_token_2` | pos_2      | Number    |
| `acceptance_length`     | accept_len | Number    |

Use a **Time series** panel. Add a **Filter by value** transformation on `test_id` to isolate individual tests, or use a dashboard variable.

### Record schema

`acceptance.json` is a JSON array of objects, each with this schema:

```json
{
  "timestamp": "2026-11-05T04:12:00+00:00",
  "git_sha": "ab2cab9...",
  "test_id": "test_eagle3_qwen3_8b_sharegpt",
  "test_file": "tests/e2e/regression/test_training_only_acceptance.py",
  "total_num_output_tokens": 512,
  "num_drafts": 100,
  "num_draft_tokens": 300,
  "num_accepted_tokens": 60,
  "acceptance_length": 1.6,
  "acceptance_at_token_0": 0.45,
  "acceptance_at_token_1": 0.12,
  "acceptance_at_token_2": 0.03
}
```

## One-time setup

Create the orphan branch (requires push access):

```bash
python scripts/ci/push_metrics.py init
```

This creates a `metrics-data` branch with no shared history.

## Local testing

```bash
# Manually push a test record
echo '{"timestamp":"2026-01-01T00:00:00Z","test_id":"manual","acceptance_at_token_0":0.5}' \
  | python scripts/ci/push_metrics.py push - --target acceptance.json
```
