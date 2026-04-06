from __future__ import annotations

from src.scripts.grounding_experiments.report_metrics import (
    compute_proximity_boxplot_groups_by_class_and_match,
    compute_proximity_histogram_distribution,
    compute_proximity_summary_by_class,
    compute_proximity_summary_by_class_match,
    compute_proximity_threshold_cumulative_coverage,
)


def _sample_records() -> list[dict[str, object]]:
    return [
        {"ground_truth_cls": "AD", "class_match": True, "proximity": 0.9},
        {"ground_truth_cls": "AD", "class_match": False, "proximity": 0.3},
        {"ground_truth_cls": "HP", "class_match": True, "proximity": 0.8},
        {"ground_truth_cls": "ASS", "class_match": False, "proximity": 0.5},
        {"ground_truth_cls": "ASS", "class_match": True, "proximity": None},
    ]


def test_proximity_histogram_distribution_counts_all_valid_values() -> None:
    result = compute_proximity_histogram_distribution(records=_sample_records(), bins=5)

    assert result["total"] == 4
    assert len(result["labels"]) == 5
    assert sum(result["counts"]) == 4


def test_proximity_summary_by_class_returns_expected_stats() -> None:
    summary = compute_proximity_summary_by_class(records=_sample_records(), target_labels=["AD", "HP", "ASS"])
    by_label = {row["label"]: row for row in summary}

    assert by_label["AD"]["count"] == 2
    assert by_label["AD"]["min"] == 0.3
    assert by_label["AD"]["max"] == 0.9
    assert by_label["HP"]["count"] == 1
    assert by_label["HP"]["mean"] == 0.8
    assert by_label["ASS"]["count"] == 1
    assert by_label["ASS"]["mean"] == 0.5


def test_proximity_summary_by_class_match_separates_cohorts() -> None:
    summary = compute_proximity_summary_by_class_match(records=_sample_records())

    assert summary["matched"]["count"] == 2
    assert summary["mismatched"]["count"] == 2


def test_proximity_boxplot_groups_contains_expected_groups() -> None:
    groups = compute_proximity_boxplot_groups_by_class_and_match(
        records=_sample_records(),
        target_labels=["AD", "HP", "ASS"],
    )
    names = [group["group"] for group in groups]

    assert "AD (acierto)" in names
    assert "AD (fallo)" in names
    assert "HP (acierto)" in names
    assert "ASS (fallo)" in names


def test_proximity_threshold_cumulative_coverage_returns_monotonic_counts() -> None:
    coverage = compute_proximity_threshold_cumulative_coverage(
        records=_sample_records(),
        thresholds=[0.3, 0.5, 0.7, 0.9],
    )

    counts = [row["count"] for row in coverage]
    assert counts == sorted(counts, reverse=True)


def test_proximity_metric_helpers_handle_empty_inputs() -> None:
    empty_records: list[dict[str, object]] = []

    histogram = compute_proximity_histogram_distribution(records=empty_records)
    summary_by_class = compute_proximity_summary_by_class(records=empty_records)
    summary_by_match = compute_proximity_summary_by_class_match(records=empty_records)
    boxplot_groups = compute_proximity_boxplot_groups_by_class_and_match(records=empty_records)
    coverage = compute_proximity_threshold_cumulative_coverage(records=empty_records)

    assert histogram["total"] == 0
    assert sum(histogram["counts"]) == 0
    assert summary_by_class[0]["count"] == 0
    assert summary_by_match["matched"]["count"] == 0
    assert summary_by_match["mismatched"]["count"] == 0
    assert boxplot_groups == []
    assert coverage == []
