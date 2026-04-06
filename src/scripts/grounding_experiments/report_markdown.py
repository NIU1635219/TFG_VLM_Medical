"""Markdown report builders for grounding scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.tests_ui.markdown_report import (
    ImageGroupSection,
    ImageItem,
    ListSection,
    SectionGroup,
    TextSection,
    write_markdown_report,
)
from src.utils.tests_ui.metrics import mean_or_none

from .report_metrics import (
    build_class_confusion_matrix,
    compute_classification_accuracy_from_records,
    compute_macro_f1_and_recall_by_class,
)
from .report_narrative import (
    build_classwise_heatmap_explanations,
    build_executive_summary_text,
    build_global_performance_explanation,
    build_metric_charts_reading_guide,
    build_metric_section_explanation,
    build_top_confusion_errors,
    describe_result_item,
)
from .report_visuals import (
    build_annotated_comparison_filename,
    write_class_confusion_heatmap,
    write_iou_boxplot_by_class_and_correctness,
    write_iou_class_summary_grouped_bars,
    write_iou_correctness_comparison_barplot,
    write_iou_distribution_histogram,
    write_iou_threshold_cumulative_curve,
    write_proximity_boxplot_by_class_and_correctness,
    write_proximity_class_summary_grouped_bars,
    write_proximity_correctness_comparison_barplot,
    write_proximity_distribution_histogram,
    write_proximity_threshold_cumulative_curve,
)


def _normalize_polyp_class(value: Any) -> str:
    return str(value or "").strip().upper()


def build_markdown_records_from_scenario_jsonl(*, output_path: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Build markdown report records from the full scenario JSONL history."""
    from .runner_core import compute_iou_safe, compute_proximity_safe, load_jsonl_records

    report_records: list[dict[str, Any]] = []
    annotated_dirs = [run_dir / "annotated", run_dir / "images"]

    for entry in load_jsonl_records(output_path):
        if not isinstance(entry, dict):
            continue

        status = str(entry.get("status") or "ok").strip().lower()
        if status not in {"ok", "success"}:
            continue

        raw_payload = entry.get("payload")
        if isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            raw_result = entry.get("result")
            payload = raw_result if isinstance(raw_result, dict) else {}

        image_id_value = entry.get("image_id")
        image_path_text = str(entry.get("image_path") or "")
        image_path_obj = Path(image_path_text) if image_path_text else Path(str(image_id_value or "image"))
        annotated_name = build_annotated_comparison_filename(
            image_id=image_id_value,
            image_path=image_path_obj,
        )
        annotated_path: str | None = None
        for base_dir in annotated_dirs:
            annotated_candidate = base_dir / annotated_name
            if annotated_candidate.exists():
                annotated_path = str(annotated_candidate)
                break

        predicted_cls = _normalize_polyp_class(
            payload.get("final_diagnosis_class")
            or entry.get("predicted_cls")
            or ""
        )
        iou_value = entry.get("iou_score")
        if not isinstance(iou_value, (int, float)):
            pred_bbox = [
                payload.get("ymin"),
                payload.get("xmin"),
                payload.get("ymax"),
                payload.get("xmax"),
            ]
            gt_bbox = entry.get("ground_truth_bbox")
            if isinstance(gt_bbox, list):
                iou_value = compute_iou_safe(gt_bbox=gt_bbox, pred_bbox=pred_bbox)
            else:
                iou_value = None

        proximity_value = entry.get("proximity_score")
        proximity_center_value = entry.get("proximity_center_score")
        proximity_size_value = entry.get("proximity_size_score")
        if not isinstance(proximity_value, (int, float)):
            pred_bbox = [
                payload.get("ymin"),
                payload.get("xmin"),
                payload.get("ymax"),
                payload.get("xmax"),
            ]
            gt_bbox = entry.get("ground_truth_bbox")
            if isinstance(gt_bbox, list):
                proximity_payload = compute_proximity_safe(gt_bbox=gt_bbox, pred_bbox=pred_bbox)
                if isinstance(proximity_payload, dict):
                    proximity_value = proximity_payload.get("proximity_score")
                    proximity_center_value = proximity_payload.get("proximity_center_score")
                    proximity_size_value = proximity_payload.get("proximity_size_score")
                else:
                    proximity_value = None
            else:
                proximity_value = None

        report_records.append(
            {
                "image_id": image_id_value,
                "image_path": image_path_text,
                "ground_truth_cls": _normalize_polyp_class(entry.get("ground_truth_cls") or ""),
                "predicted_cls": predicted_cls,
                "class_match": bool(entry.get("class_match")) if entry.get("class_match") is not None else None,
                "iou": float(iou_value) if isinstance(iou_value, (int, float)) else None,
                "proximity": float(proximity_value) if isinstance(proximity_value, (int, float)) else None,
                "proximity_center": (
                    float(proximity_center_value) if isinstance(proximity_center_value, (int, float)) else None
                ),
                "proximity_size": float(proximity_size_value) if isinstance(proximity_size_value, (int, float)) else None,
                "annotated_path": annotated_path,
            }
        )

    return report_records


def generate_single_detection_markdown_report(
    *,
    run_dir: Path,
    report_title: str,
    scenario_name: str,
    model_id: str,
    jsonl_path: Path,
    records: list[dict[str, Any]],
) -> Path:
    """Generate a markdown report for single-detection scenario outputs."""
    iou_values = [float(item["iou"]) for item in records if isinstance(item.get("iou"), (int, float))]
    avg_iou = mean_or_none(iou_values)
    proximity_values = [float(item["proximity"]) for item in records if isinstance(item.get("proximity"), (int, float))]
    avg_proximity = mean_or_none(proximity_values)
    class_metrics = compute_classification_accuracy_from_records(records=records)
    class_accuracy = class_metrics.get("accuracy")
    confusion_labels, confusion_matrix = build_class_confusion_matrix(records=records)
    class_quality = compute_macro_f1_and_recall_by_class(
        labels=confusion_labels,
        matrix=confusion_matrix,
        target_labels=["AD", "HP", "ASS"],
    )
    macro_f1 = class_quality.get("macro_f1")
    recall_by_class = class_quality.get("recall_by_class")
    top_confusions = build_top_confusion_errors(
        labels=confusion_labels,
        matrix=confusion_matrix,
        limit=5,
    )
    classwise_explanations = build_classwise_heatmap_explanations(
        labels=confusion_labels,
        matrix=confusion_matrix,
    )
    heatmap_path = write_class_confusion_heatmap(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    iou_distribution_path = write_iou_distribution_histogram(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    iou_class_summary_path = write_iou_class_summary_grouped_bars(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    iou_correctness_path = write_iou_correctness_comparison_barplot(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    iou_boxplot_path = write_iou_boxplot_by_class_and_correctness(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    iou_threshold_path = write_iou_threshold_cumulative_curve(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    proximity_distribution_path = write_proximity_distribution_histogram(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    proximity_class_summary_path = write_proximity_class_summary_grouped_bars(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    proximity_correctness_path = write_proximity_correctness_comparison_barplot(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    proximity_boxplot_path = write_proximity_boxplot_by_class_and_correctness(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    proximity_threshold_path = write_proximity_threshold_cumulative_curve(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )
    global_explanation = build_global_performance_explanation(
        accuracy=float(class_accuracy) if isinstance(class_accuracy, (int, float)) else None,
        avg_iou=float(avg_iou) if isinstance(avg_iou, (int, float)) else None,
        avg_proximity=float(avg_proximity) if isinstance(avg_proximity, (int, float)) else None,
    )
    executive_summary = build_executive_summary_text(
        total_images=len(records),
        class_accuracy=float(class_accuracy) if isinstance(class_accuracy, (int, float)) else None,
        macro_f1=float(macro_f1) if isinstance(macro_f1, (int, float)) else None,
        avg_iou=float(avg_iou) if isinstance(avg_iou, (int, float)) else None,
        avg_proximity=float(avg_proximity) if isinstance(avg_proximity, (int, float)) else None,
    )
    recall_map = recall_by_class if isinstance(recall_by_class, dict) else {}
    recall_ad = recall_map.get("AD")
    recall_hp = recall_map.get("HP")
    recall_ass = recall_map.get("ASS")
    recall_summary_text = (
        "Recall por clase (AD/HP/ASS): "
        f"AD={float(recall_ad):.2f}, "
        f"HP={float(recall_hp):.2f}, "
        f"ASS={float(recall_ass):.2f}"
        if isinstance(recall_ad, (int, float))
        and isinstance(recall_hp, (int, float))
        and isinstance(recall_ass, (int, float))
        else "Recall por clase (AD/HP/ASS): N/A"
    )

    report_sections: list[SectionGroup] = [
        SectionGroup(
            heading="Resumen global",
            heading_level=2,
            sections=[
                ListSection(
                    items=[
                        (
                            f"Accuracy global (clase): {float(class_accuracy) * 100:.2f}%. "
                            "Porcentaje de imágenes donde la clase predicha coincide con la clase real."
                            if isinstance(class_accuracy, (int, float))
                            else "Accuracy global (clase): N/A"
                        ),
                        (
                            f"Aciertos de clase: {int(class_metrics.get('matched') or 0)} casos. "
                            "Representa cuántas predicciones clasificaron correctamente la lesión."
                        ),
                        (
                            f"Errores de clase: {int(class_metrics.get('mismatched') or 0)} casos. "
                            "Casos en los que la clase predicha difiere de la clase real."
                        ),
                        (
                            f"Macro-F1 (AD/HP/ASS): {float(macro_f1):.4f}. "
                            "Resume el equilibrio global entre precisión y recall tratando todas las clases por igual."
                            if isinstance(macro_f1, (int, float))
                            else "Macro-F1 (AD/HP/ASS): N/A"
                        ),
                        recall_summary_text,
                    ],
                    ordered=False,
                    heading_level=3,
                ),
                ListSection(
                    heading="Top de errores más frecuentes",
                    heading_level=3,
                    ordered=False,
                    items=top_confusions,
                ),
                TextSection(
                    heading="Resumen ejecutivo",
                    heading_level=3,
                    text=executive_summary,
                ),
                TextSection(
                    heading="Interpretación",
                    heading_level=3,
                    text=global_explanation,
                ),
            ],
        ),
        SectionGroup(
            heading="Leyenda",
            heading_level=2,
            sections=[
                ListSection(
                    items=[
                        "GT: verde",
                        "Predicción: rojo",
                        "Texto en cajas: tipo de pólipo (AD/HP/ASS)",
                    ],
                    ordered=False,
                    heading_level=3,
                )
            ],
        )
    ]

    if heatmap_path is not None:
        report_sections.insert(
            1,
            SectionGroup(
                heading="Mapa de calor de confusión",
                heading_level=2,
                sections=[
                    ImageGroupSection(
                        images=[
                            ImageItem(
                                path=str(heatmap_path),
                                alt_text=f"{scenario_name}_class_confusion_heatmap",
                                caption="Filas = clase real (GT), columnas = clase predicha.",
                            )
                        ],
                        heading_level=3,
                    ),
                    ListSection(
                        heading="Lectura clase por clase",
                        heading_level=3,
                        ordered=False,
                        items=classwise_explanations,
                    )
                ],
            ),
        )

    iou_images: list[ImageItem] = []
    if iou_distribution_path is not None:
        iou_images.append(
            ImageItem(
                path=str(iou_distribution_path),
                alt_text=f"{scenario_name}_iou_distribution_histogram",
                caption="Distribución de puntuaciones IoU en intervalos de 0.1.",
            )
        )
    if iou_class_summary_path is not None:
        iou_images.append(
            ImageItem(
                path=str(iou_class_summary_path),
                alt_text=f"{scenario_name}_iou_class_summary_grouped_bars",
                caption="Resumen IoU por clase GT con mínimo, media y máximo.",
            )
        )
    if iou_correctness_path is not None:
        iou_images.append(
            ImageItem(
                path=str(iou_correctness_path),
                alt_text=f"{scenario_name}_iou_correctness_comparison",
                caption="Comparativa de IoU medio entre aciertos y fallos de clase.",
            )
        )
    if iou_boxplot_path is not None:
        iou_images.append(
            ImageItem(
                path=str(iou_boxplot_path),
                alt_text=f"{scenario_name}_iou_boxplot_class_correctness",
                caption="Dispersión IoU por clase y por acierto/fallo de clasificación.",
            )
        )
    if iou_threshold_path is not None:
        iou_images.append(
            ImageItem(
                path=str(iou_threshold_path),
                alt_text=f"{scenario_name}_iou_threshold_cumulative_curve",
                caption="Porcentaje de muestras que supera cada umbral IoU (>=0.3, >=0.5, >=0.7, >=0.9).",
            )
        )

    if iou_images:
        iou_sections: list[Any] = [
            TextSection(
                heading="Resumen IoU",
                heading_level=3,
                text=build_metric_section_explanation(
                    metric_name="iou",
                    average_value=float(avg_iou) if isinstance(avg_iou, (int, float)) else None,
                ),
            ),
            ImageGroupSection(
                images=iou_images,
                heading="Visualizaciones IoU",
                heading_level=3,
            ),
            ListSection(
                heading="Cómo leer los gráficos",
                heading_level=3,
                ordered=False,
                items=build_metric_charts_reading_guide(metric_name="iou"),
            ),
        ]

        iou_group = SectionGroup(
            heading="Análisis IoU",
            heading_level=2,
            sections=iou_sections,
        )
        insert_index = 2 if heatmap_path is not None else 1
        report_sections.insert(insert_index, iou_group)

    proximity_images: list[ImageItem] = []
    if proximity_distribution_path is not None:
        proximity_images.append(
            ImageItem(
                path=str(proximity_distribution_path),
                alt_text=f"{scenario_name}_proximity_distribution_histogram",
                caption="Distribución de la métrica Proximity en intervalos de 0.1.",
            )
        )
    if proximity_class_summary_path is not None:
        proximity_images.append(
            ImageItem(
                path=str(proximity_class_summary_path),
                alt_text=f"{scenario_name}_proximity_class_summary_grouped_bars",
                caption="Resumen Proximity por clase GT con mínimo, media y máximo.",
            )
        )
    if proximity_correctness_path is not None:
        proximity_images.append(
            ImageItem(
                path=str(proximity_correctness_path),
                alt_text=f"{scenario_name}_proximity_correctness_comparison",
                caption="Comparativa de Proximity medio entre aciertos y fallos de clase.",
            )
        )
    if proximity_boxplot_path is not None:
        proximity_images.append(
            ImageItem(
                path=str(proximity_boxplot_path),
                alt_text=f"{scenario_name}_proximity_boxplot_class_correctness",
                caption="Dispersión Proximity por clase y por acierto/fallo de clasificación.",
            )
        )
    if proximity_threshold_path is not None:
        proximity_images.append(
            ImageItem(
                path=str(proximity_threshold_path),
                alt_text=f"{scenario_name}_proximity_threshold_cumulative_curve",
                caption=(
                    "Porcentaje de muestras que supera cada umbral de Proximity "
                    "(>=0.3, >=0.5, >=0.7, >=0.9)."
                ),
            )
        )

    if proximity_images:
        proximity_sections: list[Any] = [
            ListSection(
                heading="Resumen Proximity",
                heading_level=3,
                ordered=False,
                items=[
                    (
                        f"Proximity medio global: {float(avg_proximity):.4f}"
                        if isinstance(avg_proximity, (int, float))
                        else "Proximity medio global: N/A"
                    ),
                    (
                        "Lectura conjunta: IoU y Proximity miden aspectos geométricos distintos y deben interpretarse de forma complementaria."
                    ),
                ],
            ),
            TextSection(
                heading="Interpretación Proximity",
                heading_level=3,
                text=build_metric_section_explanation(
                    metric_name="proximity",
                    average_value=float(avg_proximity) if isinstance(avg_proximity, (int, float)) else None,
                ),
            ),
            ImageGroupSection(
                images=proximity_images,
                heading="Visualizaciones Proximity",
                heading_level=3,
            ),
            ListSection(
                heading="Cómo leer los gráficos",
                heading_level=3,
                ordered=False,
                items=build_metric_charts_reading_guide(metric_name="proximity"),
            ),
        ]

        proximity_group = SectionGroup(
            heading="Análisis Proximity",
            heading_level=2,
            sections=proximity_sections,
        )
        if iou_images:
            insert_index = 3 if heatmap_path is not None else 2
        else:
            insert_index = 2 if heatmap_path is not None else 1
        report_sections.insert(insert_index, proximity_group)

    for index, item in enumerate(records, start=1):
        item_iou = item.get("iou")
        item_iou_value = float(item_iou) if isinstance(item_iou, (int, float)) else None
        item_proximity = item.get("proximity")
        item_proximity_value = float(item_proximity) if isinstance(item_proximity, (int, float)) else None
        summary_lines = [
            f"image_id: {item.get('image_id')}",
            f"GT class: {item.get('ground_truth_cls', 'N/D')}",
            f"Pred class: {item.get('predicted_cls', 'N/D')}",
            f"Class match: {'yes' if item.get('class_match') else 'no'}",
            (
                f"IoU: {float(item['iou']):.4f}"
                if isinstance(item.get("iou"), (int, float))
                else "IoU: N/D"
            ),
            (
                f"Proximity: {float(item['proximity']):.4f}"
                if isinstance(item.get("proximity"), (int, float))
                else "Proximity: N/D"
            ),
        ]

        sections: list[Any] = [
            ListSection(items=summary_lines, heading="Resumen", ordered=False, heading_level=3),
            TextSection(
                heading="Interpretación del caso",
                heading_level=3,
                text=describe_result_item(
                    class_match=item.get("class_match") if isinstance(item.get("class_match"), bool) else None,
                    iou_value=item_iou_value,
                    proximity_value=item_proximity_value,
                ),
            ),
        ]

        image_items: list[ImageItem] = []
        if item.get("annotated_path"):
            image_items.append(
                ImageItem(
                    path=str(item["annotated_path"]),
                    alt_text=f"{scenario_name}_{index}",
                    caption="GT (verde) vs Predicción (rojo)",
                )
            )
        if image_items:
            sections.append(ImageGroupSection(images=image_items, heading="Visualización", heading_level=3))

        report_sections.append(
            SectionGroup(
                heading=f"Resultado {index}: image_id={item.get('image_id')}",
                heading_level=2,
                sections=sections,
            )
        )

    report_path = run_dir / "results.md"
    write_markdown_report(
        report_path=report_path,
        title=report_title,
        metadata={
            "Scenario": scenario_name,
            "Model": model_id,
            "Total images": str(len(records)),
            "JSONL": jsonl_path.name,
            "Global Accuracy (class)": (
                f"{float(class_accuracy) * 100:.2f}%"
                if isinstance(class_accuracy, (int, float))
                else "N/A"
            ),
            "Macro-F1 (AD/HP/ASS)": (
                f"{float(macro_f1):.4f}"
                if isinstance(macro_f1, (int, float))
                else "N/A"
            ),
            "Global IoU (avg)": f"{avg_iou:.4f}" if avg_iou is not None else "N/A",
            "Global Proximity (avg)": f"{avg_proximity:.4f}" if avg_proximity is not None else "N/A",
        },
        sections=report_sections,
    )
    return report_path
