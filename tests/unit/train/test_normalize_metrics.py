"""Unit tests for normalize_counted_metrics."""

from speculators.train.utils import normalize_counted_metrics


class TestNormalizeCountedMetrics:
    def test_sum_total_pair_computed(self):
        metrics = {"loss_sum": 10.0, "loss_total": 5.0}
        result = normalize_counted_metrics(metrics)
        assert result["loss"] == 2.0
        assert "loss_sum" not in result
        assert "loss_total" not in result

    def test_sum_total_pair_zero_total(self):
        metrics = {"loss_sum": 10.0, "loss_total": 0.0}
        result = normalize_counted_metrics(metrics)
        assert result["loss"] == 0.0

    def test_unpaired_metric_divided_by_world_size(self):
        metrics = {"some_metric": 8.0}
        result = normalize_counted_metrics(metrics, world_size=4)
        assert result["some_metric"] == 2.0

    def test_unpaired_metric_not_divided_when_world_size_1(self):
        metrics = {"some_metric": 8.0}
        result = normalize_counted_metrics(metrics, world_size=1)
        assert result["some_metric"] == 8.0

    def test_paired_metrics_not_divided_by_world_size(self):
        metrics = {"loss_sum": 10.0, "loss_total": 5.0}
        result = normalize_counted_metrics(metrics, world_size=4)
        # Paired metrics use sum/total ratio, world_size should not affect them
        assert result["loss"] == 2.0

    def test_mixed_paired_and_unpaired(self):
        metrics = {
            "loss_sum": 10.0,
            "loss_total": 5.0,
            "eal_sum": 6.0,
            "eal_total": 3.0,
            "standalone": 12.0,
        }
        result = normalize_counted_metrics(metrics, world_size=3)
        assert result["loss"] == 2.0
        assert result["eal"] == 2.0
        assert result["standalone"] == 4.0

    def test_multiple_sum_total_pairs(self):
        metrics = {
            "loss_sum": 10.0,
            "loss_total": 2.0,
            "acc_sum": 8.0,
            "acc_total": 10.0,
            "position_0_acc_sum": 5.0,
            "position_0_acc_total": 10.0,
        }
        result = normalize_counted_metrics(metrics)
        assert result["loss"] == 5.0
        assert result["acc"] == 0.8
        assert result["position_0_acc"] == 0.5

    def test_total_without_matching_sum_is_removed(self):
        metrics = {"orphan_total": 5.0, "other": 3.0}
        result = normalize_counted_metrics(metrics)
        assert "orphan_total" not in result
        assert result["other"] == 3.0

    def test_empty_dict(self):
        result = normalize_counted_metrics({})
        assert result == {}
