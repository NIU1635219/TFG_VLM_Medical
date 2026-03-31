"""Visual helpers for grounding reports and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.tests_ui.visualizer import draw_comparison_bboxes

from .report_metrics import build_class_confusion_matrix


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
