"""Metrics helpers for grounding markdown reports.

This module centralizes report-level metrics and matrix computations so
runner_core can stay focused on orchestration and I/O.
"""

from __future__ import annotations

import math
from typing import Any

from src.utils.tests_ui.metrics import mean_or_none


def _normalize_class(value: Any) -> str:
    """Normalize class labels to stable uppercase text."""
    return str(value or "").strip().upper()


def _extract_iou_rows(*, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract IoU-ready rows with normalized class and class_match metadata."""
    rows: list[dict[str, Any]] = []
    for item in records:
        raw_iou = item.get("iou")
        if not isinstance(raw_iou, (int, float)):
            continue
        iou_value = float(raw_iou)
        if math.isnan(iou_value) or math.isinf(iou_value):
            continue
        if iou_value < 0.0:
            iou_value = 0.0
        if iou_value > 1.0:
            iou_value = 1.0

        raw_match = item.get("class_match")
        match_value = raw_match if isinstance(raw_match, bool) else None
        rows.append(
            {
                "iou": iou_value,
                "ground_truth_cls": _normalize_class(item.get("ground_truth_cls") or ""),
                "class_match": match_value,
            }
        )
    return rows


def _extract_metric_rows(*, records: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    """Extract metric-ready rows with normalized class and class_match metadata."""
    rows: list[dict[str, Any]] = []
    for item in records:
        raw_value = item.get(metric_key)
        if not isinstance(raw_value, (int, float)):
            continue
        metric_value = float(raw_value)
        if math.isnan(metric_value) or math.isinf(metric_value):
            continue
        if metric_value < 0.0:
            metric_value = 0.0
        if metric_value > 1.0:
            metric_value = 1.0

        raw_match = item.get("class_match")
        match_value = raw_match if isinstance(raw_match, bool) else None
        rows.append(
            {
                "value": metric_value,
                "ground_truth_cls": _normalize_class(item.get("ground_truth_cls") or ""),
                "class_match": match_value,
            }
        )
    return rows


def _extract_proximity_rows(*, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract proximity-ready rows with normalized class and class_match metadata."""
    return _extract_metric_rows(records=records, metric_key="proximity")


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    """Return stable summary stats for a numeric list."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean_or_none(values),
        "min": min(values),
        "max": max(values),
    }


def compute_classification_accuracy_from_records(*, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute class-level global accuracy metrics from report-ready records."""
    compared = 0
    matched = 0

    for item in records:
        gt_cls = _normalize_class(item.get("ground_truth_cls") or "")
        pred_cls = _normalize_class(item.get("predicted_cls") or "")
        if not gt_cls or not pred_cls:
            continue
        compared += 1
        if gt_cls == pred_cls:
            matched += 1

    mismatched = max(0, compared - matched)
    accuracy = (matched / compared) if compared > 0 else None
    return {
        "compared": compared,
        "matched": matched,
        "mismatched": mismatched,
        "accuracy": accuracy,
    }


def build_class_confusion_matrix(*, records: list[dict[str, Any]]) -> tuple[list[str], list[list[int]]]:
    """Build confusion matrix (GT rows vs Pred columns) from report-ready records."""
    base_labels = ["AD", "HP", "ASS"]
    pairs: list[tuple[str, str]] = []

    for item in records:
        gt_cls = _normalize_class(item.get("ground_truth_cls") or "")
        pred_cls = _normalize_class(item.get("predicted_cls") or "")
        if gt_cls and pred_cls:
            pairs.append((gt_cls, pred_cls))

    if not pairs:
        return [], []

    base_set = set(base_labels)
    extra_labels = sorted(
        {
            label
            for pair in pairs
            for label in pair
            if label not in base_set
        }
    )
    labels = base_labels + extra_labels
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for gt_cls, pred_cls in pairs:
        row = index_by_label.get(gt_cls)
        col = index_by_label.get(pred_cls)
        if row is None or col is None:
            continue
        matrix[row][col] += 1

    return labels, matrix


def compute_macro_f1_and_recall_by_class(
    *,
    labels: list[str],
    matrix: list[list[int]],
    target_labels: list[str],
) -> dict[str, Any]:
    """Compute recall by class and macro-F1 for selected labels from confusion matrix."""
    label_to_index = {label: index for index, label in enumerate(labels)}
    recalls: dict[str, float | None] = {}
    f1_values: list[float] = []

    for label in target_labels:
        index = label_to_index.get(label)
        if index is None or index >= len(matrix):
            recalls[label] = None
            continue

        row = matrix[index]
        tp = float(row[index]) if index < len(row) else 0.0
        support = float(sum(int(value) for value in row))
        predicted_total = float(
            sum(int(matrix[row_idx][index]) for row_idx in range(len(matrix)) if index < len(matrix[row_idx]))
        )

        recall = (tp / support) if support > 0 else None
        precision = (tp / predicted_total) if predicted_total > 0 else None
        recalls[label] = recall

        if recall is not None and precision is not None and (precision + recall) > 0:
            f1_values.append(2.0 * precision * recall / (precision + recall))

    macro_f1 = mean_or_none(f1_values)
    return {
        "macro_f1": macro_f1,
        "recall_by_class": recalls,
    }


def compute_iou_histogram_distribution(
    *,
    records: list[dict[str, Any]],
    bins: int = 10,
) -> dict[str, Any]:
    """Compute a fixed-width IoU histogram in range [0, 1]."""
    rows = _extract_iou_rows(records=records)
    if bins <= 0:
        bins = 10

    counts = [0 for _ in range(bins)]
    labels: list[str] = []
    step = 1.0 / float(bins)
    for index in range(bins):
        left = step * index
        right = step * (index + 1)
        if index == bins - 1:
            labels.append(f"[{left:.1f}, {right:.1f}]")
        else:
            labels.append(f"[{left:.1f}, {right:.1f})")

    for row in rows:
        iou_value = float(row["iou"])
        bin_index = min(int(iou_value * bins), bins - 1)
        counts[bin_index] += 1

    return {
        "labels": labels,
        "counts": counts,
        "total": len(rows),
    }


def compute_iou_summary_by_class(
    *,
    records: list[dict[str, Any]],
    target_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute IoU summary stats by ground-truth class."""
    rows = _extract_iou_rows(records=records)
    normalized_targets = [_normalize_class(label) for label in (target_labels or ["AD", "HP", "ASS"]) if label]

    values_by_class: dict[str, list[float]] = {label: [] for label in normalized_targets}
    for row in rows:
        class_name = _normalize_class(row.get("ground_truth_cls") or "")
        if not class_name:
            continue
        values_by_class.setdefault(class_name, []).append(float(row["iou"]))

    ordered_labels = normalized_targets + sorted(
        label for label in values_by_class.keys() if label not in set(normalized_targets)
    )

    result: list[dict[str, Any]] = []
    for label in ordered_labels:
        stats = _summary_stats(values_by_class.get(label, []))
        count_value = stats.get("count")
        count_int = int(count_value) if isinstance(count_value, (int, float)) else 0
        result.append(
            {
                "label": label,
                "count": count_int,
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"],
            }
        )
    return result


def compute_iou_summary_by_class_match(*, records: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """Compute IoU summary for class-match true/false cohorts."""
    rows = _extract_iou_rows(records=records)
    matched_values = [float(row["iou"]) for row in rows if row.get("class_match") is True]
    mismatched_values = [float(row["iou"]) for row in rows if row.get("class_match") is False]

    return {
        "matched": _summary_stats(matched_values),
        "mismatched": _summary_stats(mismatched_values),
    }


def compute_iou_boxplot_groups_by_class_and_match(
    *,
    records: list[dict[str, Any]],
    target_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare grouped IoU values for boxplots split by class and class_match."""
    rows = _extract_iou_rows(records=records)
    normalized_targets = [_normalize_class(label) for label in (target_labels or ["AD", "HP", "ASS"]) if label]

    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        class_name = _normalize_class(row.get("ground_truth_cls") or "")
        if not class_name:
            continue
        class_match = row.get("class_match")
        if class_match is True:
            status = "acierto"
        elif class_match is False:
            status = "fallo"
        else:
            status = "sin_dato"
        groups.setdefault((class_name, status), []).append(float(row["iou"]))

    extra_labels = sorted(
        label for label, _ in groups.keys() if label not in set(normalized_targets)
    )
    ordered_classes = normalized_targets + extra_labels
    ordered_statuses = ["acierto", "fallo", "sin_dato"]

    result: list[dict[str, Any]] = []
    for class_name in ordered_classes:
        for status in ordered_statuses:
            values = groups.get((class_name, status), [])
            if not values:
                continue
            result.append(
                {
                    "group": f"{class_name} ({status})",
                    "class_name": class_name,
                    "status": status,
                    "values": values,
                }
            )
    return result


def compute_iou_threshold_cumulative_coverage(
    *,
    records: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """Compute percentage of IoU samples above fixed thresholds."""
    rows = _extract_iou_rows(records=records)
    values = [float(row["iou"]) for row in rows]
    if not values:
        return []

    chosen_thresholds = thresholds or [0.3, 0.5, 0.7, 0.9]
    normalized_thresholds = sorted(
        {
            min(1.0, max(0.0, float(value)))
            for value in chosen_thresholds
            if isinstance(value, (int, float))
        }
    )

    total = len(values)
    result: list[dict[str, float | int]] = []
    for threshold in normalized_thresholds:
        covered = sum(1 for value in values if value >= threshold)
        result.append(
            {
                "threshold": threshold,
                "count": covered,
                "ratio": (covered / total) if total > 0 else 0.0,
            }
        )
    return result


def compute_proximity_histogram_distribution(
    *,
    records: list[dict[str, Any]],
    bins: int = 10,
) -> dict[str, Any]:
    """Compute a fixed-width proximity histogram in range [0, 1]."""
    rows = _extract_proximity_rows(records=records)
    if bins <= 0:
        bins = 10

    counts = [0 for _ in range(bins)]
    labels: list[str] = []
    step = 1.0 / float(bins)
    for index in range(bins):
        left = step * index
        right = step * (index + 1)
        if index == bins - 1:
            labels.append(f"[{left:.1f}, {right:.1f}]")
        else:
            labels.append(f"[{left:.1f}, {right:.1f})")

    for row in rows:
        value = float(row["value"])
        bin_index = min(int(value * bins), bins - 1)
        counts[bin_index] += 1

    return {
        "labels": labels,
        "counts": counts,
        "total": len(rows),
    }


def compute_proximity_summary_by_class(
    *,
    records: list[dict[str, Any]],
    target_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute proximity summary stats by ground-truth class."""
    rows = _extract_proximity_rows(records=records)
    normalized_targets = [_normalize_class(label) for label in (target_labels or ["AD", "HP", "ASS"]) if label]

    values_by_class: dict[str, list[float]] = {label: [] for label in normalized_targets}
    for row in rows:
        class_name = _normalize_class(row.get("ground_truth_cls") or "")
        if not class_name:
            continue
        values_by_class.setdefault(class_name, []).append(float(row["value"]))

    ordered_labels = normalized_targets + sorted(
        label for label in values_by_class.keys() if label not in set(normalized_targets)
    )

    result: list[dict[str, Any]] = []
    for label in ordered_labels:
        stats = _summary_stats(values_by_class.get(label, []))
        count_value = stats.get("count")
        count_int = int(count_value) if isinstance(count_value, (int, float)) else 0
        result.append(
            {
                "label": label,
                "count": count_int,
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"],
            }
        )
    return result


def compute_proximity_summary_by_class_match(
    *, records: list[dict[str, Any]]
) -> dict[str, dict[str, float | int | None]]:
    """Compute proximity summary for class-match true/false cohorts."""
    rows = _extract_proximity_rows(records=records)
    matched_values = [float(row["value"]) for row in rows if row.get("class_match") is True]
    mismatched_values = [float(row["value"]) for row in rows if row.get("class_match") is False]

    return {
        "matched": _summary_stats(matched_values),
        "mismatched": _summary_stats(mismatched_values),
    }


def compute_proximity_boxplot_groups_by_class_and_match(
    *,
    records: list[dict[str, Any]],
    target_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare grouped proximity values for boxplots split by class and class_match."""
    rows = _extract_proximity_rows(records=records)
    normalized_targets = [_normalize_class(label) for label in (target_labels or ["AD", "HP", "ASS"]) if label]

    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        class_name = _normalize_class(row.get("ground_truth_cls") or "")
        if not class_name:
            continue
        class_match = row.get("class_match")
        if class_match is True:
            status = "acierto"
        elif class_match is False:
            status = "fallo"
        else:
            status = "sin_dato"
        groups.setdefault((class_name, status), []).append(float(row["value"]))

    extra_labels = sorted(
        label for label, _ in groups.keys() if label not in set(normalized_targets)
    )
    ordered_classes = normalized_targets + extra_labels
    ordered_statuses = ["acierto", "fallo", "sin_dato"]

    result: list[dict[str, Any]] = []
    for class_name in ordered_classes:
        for status in ordered_statuses:
            values = groups.get((class_name, status), [])
            if not values:
                continue
            result.append(
                {
                    "group": f"{class_name} ({status})",
                    "class_name": class_name,
                    "status": status,
                    "values": values,
                }
            )
    return result


def compute_proximity_threshold_cumulative_coverage(
    *,
    records: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """Compute percentage of proximity samples above fixed thresholds."""
    rows = _extract_proximity_rows(records=records)
    values = [float(row["value"]) for row in rows]
    if not values:
        return []

    chosen_thresholds = thresholds or [0.3, 0.5, 0.7, 0.9]
    normalized_thresholds = sorted(
        {
            min(1.0, max(0.0, float(value)))
            for value in chosen_thresholds
            if isinstance(value, (int, float))
        }
    )

    total = len(values)
    result: list[dict[str, float | int]] = []
    for threshold in normalized_thresholds:
        covered = sum(1 for value in values if value >= threshold)
        result.append(
            {
                "threshold": threshold,
                "count": covered,
                "ratio": (covered / total) if total > 0 else 0.0,
            }
        )
    return result
