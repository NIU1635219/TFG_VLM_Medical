"""Metrics helpers for grounding markdown reports.

This module centralizes report-level metrics and matrix computations so
runner_core can stay focused on orchestration and I/O.
"""

from __future__ import annotations

from typing import Any

from src.utils.tests_ui.metrics import mean_or_none


def _normalize_class(value: Any) -> str:
    """Normalize class labels to stable uppercase text."""
    return str(value or "").strip().upper()


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
