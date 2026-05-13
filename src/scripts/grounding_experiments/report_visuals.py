"""Visual helpers for grounding reports and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.tests_ui.visualizer import draw_comparison_bboxes

from .report_metrics import (
    build_class_confusion_matrix,
    compute_iou_boxplot_groups_by_class_and_match,
    compute_iou_histogram_distribution,
    compute_iou_summary_by_class,
    compute_iou_summary_by_class_match,
    compute_iou_threshold_cumulative_coverage,
    compute_proximity_boxplot_groups_by_class_and_match,
    compute_proximity_histogram_distribution,
    compute_proximity_summary_by_class,
    compute_proximity_summary_by_class_match,
    compute_proximity_threshold_cumulative_coverage,
)


def _normalize_image_stem(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return text


def _normalize_bbox_for_image(values: list[Any]) -> list[int] | None:
    if len(values) != 4:
        return None
    normalized: list[int] = []
    for value in values:
        try:
            normalized.append(int(round(float(value))))
        except Exception:
            return None
    return normalized


def build_annotated_comparison_filename(*, image_id: Any, image_path: Path) -> str:
    """Build a deterministic annotated image name for one scenario record."""
    stem = image_path.stem.strip() or "image"
    normalized_id = _normalize_image_stem(image_id)
    safe_id = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in normalized_id)
    if not safe_id:
        safe_id = "unknown"
    return f"{stem}__{safe_id}_comparison.jpg"


def write_comparison_image(
    *,
    image_path: Path,
    output_path: Path,
    gt_bbox: list[Any],
    pred_bbox: list[Any],
    model_name: str,
    gt_label: str,
    pred_label: str,
    iou_score: float | None,
) -> Path | None:
    """Generate GT vs prediction comparison image for a single detection scenario."""
    gt_norm = _normalize_bbox_for_image(gt_bbox)
    pred_norm = _normalize_bbox_for_image(pred_bbox)
    if gt_norm is None or pred_norm is None:
        return None

    try:
        draw_comparison_bboxes(
            image_path=str(image_path),
            gt_bbox=gt_norm,
            pred_bbox=pred_norm,
            model_name=model_name,
            iou_score=iou_score,
            gt_label=gt_label,
            pred_label=pred_label,
            show_overlay_text=False,
            output_path=str(output_path),
        )
        return output_path
    except Exception:
        return None


def write_class_confusion_heatmap(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate and persist a confusion heatmap image (GT rows vs Pred columns)."""
    labels, matrix = build_class_confusion_matrix(records=records)
    if not labels:
        return None

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "class_confusion_heatmap.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=axis,
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Ground-truth class")
        axis.set_title(f"{scenario_name} · Confusion Heatmap")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_iou_distribution_histogram(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate IoU distribution histogram in report_assets."""
    histogram = compute_iou_histogram_distribution(records=records, bins=10)
    labels = histogram.get("labels")
    counts = histogram.get("counts")
    total = int(histogram.get("total") or 0)
    if not isinstance(labels, list) or not isinstance(counts, list) or total <= 0:
        return None

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "iou_distribution_histogram.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions = list(range(len(labels)))
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(positions, counts, color="#2878B5", alpha=0.9)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylabel("Número de muestras")
        axis.set_xlabel("Rango IoU")
        axis.set_title(f"{scenario_name} · Distribución de IoU (n={total})")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_iou_class_summary_grouped_bars(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate grouped bars with mean/min/max IoU by class."""
    rows = compute_iou_summary_by_class(records=records, target_labels=["AD", "HP", "ASS"])
    usable_rows = [row for row in rows if isinstance(row.get("mean"), (int, float))]
    if not usable_rows:
        return None

    labels = [str(row.get("label") or "") for row in usable_rows]
    mean_values = [float(row.get("mean") or 0.0) for row in usable_rows]
    min_values = [float(row.get("min") or 0.0) for row in usable_rows]
    max_values = [float(row.get("max") or 0.0) for row in usable_rows]

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "iou_class_summary_grouped_bars.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions = list(range(len(labels)))
        width = 0.24
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar([pos - width for pos in positions], min_values, width=width, label="Mínimo", color="#63ACBE")
        axis.bar(positions, mean_values, width=width, label="Media", color="#C7A76C")
        axis.bar([pos + width for pos in positions], max_values, width=width, label="Máximo", color="#EE442F")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("IoU")
        axis.set_xlabel("Clase GT")
        axis.set_title(f"{scenario_name} · IoU por clase (mín/media/máx)")
        axis.legend(loc="upper right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_iou_correctness_comparison_barplot(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate mean IoU comparison for class-match cohorts."""
    summary = compute_iou_summary_by_class_match(records=records)
    matched_raw = summary.get("matched")
    mismatched_raw = summary.get("mismatched")
    matched: dict[str, float | int | None] = matched_raw if isinstance(matched_raw, dict) else {}
    mismatched: dict[str, float | int | None] = mismatched_raw if isinstance(mismatched_raw, dict) else {}

    labels: list[str] = []
    values: list[float] = []
    counts: list[int] = []
    matched_mean = matched.get("mean")
    if isinstance(matched_mean, (int, float)):
        labels.append("Acierto de clase")
        values.append(float(matched_mean))
        counts.append(int(matched.get("count") or 0))
    mismatched_mean = mismatched.get("mean")
    if isinstance(mismatched_mean, (int, float)):
        labels.append("Fallo de clase")
        values.append(float(mismatched_mean))
        counts.append(int(mismatched.get("count") or 0))
    if not labels:
        return None

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "iou_correctness_comparison.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(6.2, 4.6))
        bars = axis.bar(labels, values, color=["#228B22", "#CD5C5C"][: len(labels)], alpha=0.88)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("IoU medio")
        axis.set_title(f"{scenario_name} · IoU según acierto de clase")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        for index, bar in enumerate(bars):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"n={counts[index]}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_iou_boxplot_by_class_and_correctness(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate IoU boxplot split by class and correctness."""
    groups = compute_iou_boxplot_groups_by_class_and_match(records=records, target_labels=["AD", "HP", "ASS"])
    if not groups:
        return None

    labels = [str(group.get("group") or "") for group in groups]
    values = [list(group.get("values") or []) for group in groups]

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "iou_boxplot_class_correctness.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(max(7.6, len(labels) * 1.2), 4.8))
        axis.boxplot(values, tick_labels=labels, patch_artist=True)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("IoU")
        axis.set_xlabel("Clase GT y estado de clasificación")
        axis.set_title(f"{scenario_name} · Distribución IoU por clase/acierto")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_iou_threshold_cumulative_curve(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate cumulative IoU-threshold coverage chart."""
    coverage = compute_iou_threshold_cumulative_coverage(records=records, thresholds=[0.3, 0.5, 0.7, 0.9])
    if not coverage:
        return None

    thresholds = [float(item["threshold"]) for item in coverage]
    percentages = [float(item["ratio"]) * 100.0 for item in coverage]

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "iou_threshold_cumulative_curve.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(6.8, 4.6))
        axis.plot(thresholds, percentages, marker="o", linewidth=2.0, color="#7A5195")
        axis.set_xticks(thresholds)
        axis.set_xticklabels([f">={threshold:.1f}" for threshold in thresholds])
        axis.set_ylim(0.0, 100.0)
        axis.set_ylabel("Cobertura acumulada (%)")
        axis.set_xlabel("Umbral de IoU")
        axis.set_title(f"{scenario_name} · Cobertura por umbral IoU")
        axis.grid(axis="both", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_proximity_distribution_histogram(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate proximity distribution histogram in report_assets."""
    histogram = compute_proximity_histogram_distribution(records=records, bins=10)
    labels = histogram.get("labels")
    counts = histogram.get("counts")
    total = int(histogram.get("total") or 0)
    if not isinstance(labels, list) or not isinstance(counts, list) or total <= 0:
        return None

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "proximity_distribution_histogram.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions = list(range(len(labels)))
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(positions, counts, color="#2C7FB8", alpha=0.9)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylabel("Número de muestras")
        axis.set_xlabel("Rango Proximity")
        axis.set_title(f"{scenario_name} · Distribución de Proximity (n={total})")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_proximity_class_summary_grouped_bars(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate grouped bars with mean/min/max proximity by class."""
    rows = compute_proximity_summary_by_class(records=records, target_labels=["AD", "HP", "ASS"])
    usable_rows = [row for row in rows if isinstance(row.get("mean"), (int, float))]
    if not usable_rows:
        return None

    labels = [str(row.get("label") or "") for row in usable_rows]
    mean_values = [float(row.get("mean") or 0.0) for row in usable_rows]
    min_values = [float(row.get("min") or 0.0) for row in usable_rows]
    max_values = [float(row.get("max") or 0.0) for row in usable_rows]

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "proximity_class_summary_grouped_bars.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions = list(range(len(labels)))
        width = 0.24
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar([pos - width for pos in positions], min_values, width=width, label="Mínimo", color="#63ACBE")
        axis.bar(positions, mean_values, width=width, label="Media", color="#C7A76C")
        axis.bar([pos + width for pos in positions], max_values, width=width, label="Máximo", color="#EE442F")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Proximity")
        axis.set_xlabel("Clase GT")
        axis.set_title(f"{scenario_name} · Proximity por clase (mín/media/máx)")
        axis.legend(loc="upper right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_proximity_correctness_comparison_barplot(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate mean proximity comparison for class-match cohorts."""
    summary = compute_proximity_summary_by_class_match(records=records)
    matched_raw = summary.get("matched")
    mismatched_raw = summary.get("mismatched")
    matched: dict[str, float | int | None] = matched_raw if isinstance(matched_raw, dict) else {}
    mismatched: dict[str, float | int | None] = mismatched_raw if isinstance(mismatched_raw, dict) else {}

    labels: list[str] = []
    values: list[float] = []
    counts: list[int] = []
    matched_mean = matched.get("mean")
    if isinstance(matched_mean, (int, float)):
        labels.append("Acierto de clase")
        values.append(float(matched_mean))
        counts.append(int(matched.get("count") or 0))
    mismatched_mean = mismatched.get("mean")
    if isinstance(mismatched_mean, (int, float)):
        labels.append("Fallo de clase")
        values.append(float(mismatched_mean))
        counts.append(int(mismatched.get("count") or 0))
    if not labels:
        return None

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "proximity_correctness_comparison.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(6.2, 4.6))
        bars = axis.bar(labels, values, color=["#2E8B57", "#B22222"][: len(labels)], alpha=0.88)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Proximity medio")
        axis.set_title(f"{scenario_name} · Proximity según acierto de clase")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        for index, bar in enumerate(bars):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"n={counts[index]}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_proximity_boxplot_by_class_and_correctness(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate proximity boxplot split by class and correctness."""
    groups = compute_proximity_boxplot_groups_by_class_and_match(records=records, target_labels=["AD", "HP", "ASS"])
    if not groups:
        return None

    labels = [str(group.get("group") or "") for group in groups]
    values = [list(group.get("values") or []) for group in groups]

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "proximity_boxplot_class_correctness.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(max(7.6, len(labels) * 1.2), 4.8))
        axis.boxplot(values, tick_labels=labels, patch_artist=True)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Proximity")
        axis.set_xlabel("Clase GT y estado de clasificación")
        axis.set_title(f"{scenario_name} · Distribución Proximity por clase/acierto")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_proximity_threshold_cumulative_curve(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Generate cumulative proximity-threshold coverage chart."""
    coverage = compute_proximity_threshold_cumulative_coverage(records=records, thresholds=[0.3, 0.5, 0.7, 0.9])
    if not coverage:
        return None

    thresholds = [float(item["threshold"]) for item in coverage]
    percentages = [float(item["ratio"]) * 100.0 for item in coverage]

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_path = assets_dir / "proximity_threshold_cumulative_curve.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(6.8, 4.6))
        axis.plot(thresholds, percentages, marker="o", linewidth=2.0, color="#2A9D8F")
        axis.set_xticks(thresholds)
        axis.set_xticklabels([f">={threshold:.1f}" for threshold in thresholds])
        axis.set_ylim(0.0, 100.0)
        axis.set_ylabel("Cobertura acumulada (%)")
        axis.set_xlabel("Umbral de Proximity")
        axis.set_title(f"{scenario_name} · Cobertura por umbral Proximity")
        axis.grid(axis="both", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def write_scenario_s_kpi_charts(*, run_dir: Path, kpis: dict[str, Any], level: int) -> dict[str, Path]:
    """Generate Scenario S-specific KPI charts in report_assets."""
    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return output_paths

    contradiction_rate = kpis.get("contradiction_rate")
    obedience_rate = kpis.get("obedience_rate")
    polyp_detected_rate = kpis.get("polyp_detected_rate")

    rate_labels: list[str] = []
    rate_values: list[float] = []
    for label, value in (
        ("Contradiction", contradiction_rate),
        ("Obedience", obedience_rate),
        ("Polyp detected", polyp_detected_rate),
    ):
        if isinstance(value, (int, float)):
            rate_labels.append(label)
            rate_values.append(max(0.0, min(1.0, float(value))))

    if rate_labels:
        rates_path = assets_dir / "scenario_s_kpi_rates.png"
        try:
            fig, axis = plt.subplots(figsize=(7.2, 4.6))
            bars = axis.bar(
                rate_labels,
                [value * 100.0 for value in rate_values],
                color=["#2E8B57", "#B22222", "#2878B5"],
            )
            axis.set_ylim(0.0, 100.0)
            axis.set_ylabel("Rate (%)")
            axis.set_title(f"Scenario S L{level} · KPI rates")
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            for index, bar in enumerate(bars):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.0,
                    f"{rate_values[index] * 100.0:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            fig.tight_layout()
            fig.savefig(rates_path, dpi=160)
            plt.close(fig)
            output_paths["kpi_rates"] = rates_path
        except Exception:
            plt.close("all")

    by_gt_class = kpis.get("by_gt_class")
    if isinstance(by_gt_class, dict):
        labels = ["AD", "HP", "ASS"]
        true_counts = [
            int((by_gt_class.get(label) or {}).get("TRUE") or 0)
            if isinstance(by_gt_class.get(label), dict)
            else 0
            for label in labels
        ]
        false_counts = [
            int((by_gt_class.get(label) or {}).get("FALSE") or 0)
            if isinstance(by_gt_class.get(label), dict)
            else 0
            for label in labels
        ]

        if any(value > 0 for value in true_counts + false_counts):
            by_class_path = assets_dir / "scenario_s_contradiction_by_gt_class.png"
            try:
                positions = list(range(len(labels)))
                width = 0.36
                fig, axis = plt.subplots(figsize=(8.0, 4.8))
                axis.bar(
                    [position - width / 2.0 for position in positions],
                    true_counts,
                    width=width,
                    label="TRUE (contradiction)",
                    color="#2E8B57",
                )
                axis.bar(
                    [position + width / 2.0 for position in positions],
                    false_counts,
                    width=width,
                    label="FALSE (obedience)",
                    color="#B22222",
                )
                axis.set_xticks(positions)
                axis.set_xticklabels(labels)
                axis.set_ylabel("Count")
                axis.set_xlabel("GT class")
                axis.set_title(f"Scenario S L{level} · Contradiction by GT class")
                axis.legend(loc="upper right")
                axis.grid(axis="y", linestyle="--", alpha=0.35)
                fig.tight_layout()
                fig.savefig(by_class_path, dpi=160)
                plt.close(fig)
                output_paths["by_gt_class"] = by_class_path
            except Exception:
                plt.close("all")

        # Also create a percentage-stacked bar per GT class (TRUE vs FALSE % within class)
        try:
            percent_path = assets_dir / "scenario_s_contradiction_by_gt_class_percent.png"
            fig, axis = plt.subplots(figsize=(8.0, 4.8))
            class_perc_true = []
            class_perc_false = []
            for t, f in zip(true_counts, false_counts):
                total = float(t + f) if (t + f) > 0 else 0.0
                class_perc_true.append((float(t) / total * 100.0) if total > 0 else 0.0)
                class_perc_false.append((float(f) / total * 100.0) if total > 0 else 0.0)

            positions = list(range(len(labels)))
            axis.bar(positions, class_perc_true, width=0.6, label="TRUE % (contradiction)", color="#2E8B57")
            axis.bar(positions, class_perc_false, width=0.6, bottom=class_perc_true, label="FALSE % (obedience)", color="#B22222")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels)
            axis.set_ylim(0.0, 100.0)
            axis.set_ylabel("% within GT class")
            axis.set_xlabel("GT class")
            axis.set_title(f"Scenario S L{level} · % Contradiction vs Obedience by GT class")
            axis.legend(loc="upper right")
            for idx, (t_pct, f_pct) in enumerate(zip(class_perc_true, class_perc_false)):
                axis.text(idx, t_pct / 2.0, f"{t_pct:.1f}%", ha="center", va="center", color="white", fontsize=9)
                axis.text(idx, t_pct + f_pct / 2.0, f"{f_pct:.1f}%", ha="center", va="center", color="white", fontsize=9)
            fig.tight_layout()
            fig.savefig(percent_path, dpi=160)
            plt.close(fig)
            output_paths["by_gt_class_percent"] = percent_path
        except Exception:
            plt.close("all")

        if any(value > 0 for value in true_counts + false_counts):
            heatmap_path = assets_dir / "scenario_s_polyp_type_heatmap.png"
            try:
                import seaborn as sns

                matrix = [[true_counts[index], false_counts[index]] for index in range(len(labels))]
                fig, axis = plt.subplots(figsize=(7.6, 4.9))
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="YlGnBu",
                    xticklabels=["Contradicción", "Obediencia"],
                    yticklabels=labels,
                    cbar=True,
                    linewidths=0.5,
                    linecolor="#F0F0F0",
                    ax=axis,
                )
                axis.set_xlabel("Resultado frente al prompt")
                axis.set_ylabel("Tipo de pólipo (GT)")
                axis.set_title(f"Scenario S L{level} · Heatmap por tipo de pólipo")
                fig.tight_layout()
                fig.savefig(heatmap_path, dpi=160)
                plt.close(fig)
                output_paths["polyp_type_heatmap"] = heatmap_path
            except Exception:
                plt.close("all")

    return output_paths
