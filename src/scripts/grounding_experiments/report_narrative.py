"""Narrative helpers for grounding markdown reports.

This module centralizes user-facing explanations, qualitative comparisons,
and readable text generation so runner_core stays focused on data handling.
"""

from __future__ import annotations


def _normalize_metric_name(metric_name: str) -> str:
    normalized = str(metric_name or "").strip().lower()
    if normalized in {"iou", "intersection over union"}:
        return "iou"
    if normalized in {"proximity", "prox"}:
        return "proximity"
    return normalized


def build_global_performance_explanation(
    *,
    accuracy: float | None,
    avg_iou: float | None,
    avg_proximity: float | None = None,
) -> str:
    """Build a generic explanation that varies by simple comparison rules."""
    if accuracy is None and avg_iou is None and avg_proximity is None:
        return (
            "No hay suficientes datos para interpretar el rendimiento global. "
            "Se necesitan clases GT y predichas validas para estimar accuracy, IoU y Proximity."
        )

    if accuracy is None:
        return (
            "No se pudo calcular accuracy global de clase, pero si hay metrica espacial disponible. "
            "Revisa que todas las predicciones incluyan clase para poder comparar clasificacion."
        )

    if avg_iou is None and avg_proximity is None:
        if accuracy > 0.7:
            return (
                "El accuracy global es alto (> 70%), lo que sugiere buena clasificacion. "
                "No hay IoU/Proximity medio para evaluar la precision espacial de las cajas."
            )
        if accuracy < 0.4:
            return (
                "El accuracy global es bajo (< 40%), indicando errores frecuentes de clase. "
                "No hay IoU/Proximity medio para contrastar el componente espacial."
            )
        return (
            "El accuracy global es intermedio (=~ rango medio), con margen de mejora en clasificacion. "
            "No hay IoU/Proximity medio para completar el analisis espacial."
        )

    metric_relations: list[str] = []
    if isinstance(avg_iou, (int, float)):
        if accuracy > avg_iou:
            metric_relations.append("accuracy > IoU")
        elif accuracy < avg_iou:
            metric_relations.append("accuracy < IoU")
        else:
            metric_relations.append("accuracy = IoU")
    if isinstance(avg_proximity, (int, float)):
        if accuracy > avg_proximity:
            metric_relations.append("accuracy > Proximity")
        elif accuracy < avg_proximity:
            metric_relations.append("accuracy < Proximity")
        else:
            metric_relations.append("accuracy = Proximity")
    relation = ", ".join(metric_relations) if metric_relations else "comparativa no disponible"

    spatial_low = False
    if isinstance(avg_iou, (int, float)) and avg_iou < 0.2:
        spatial_low = True
    if isinstance(avg_proximity, (int, float)) and avg_proximity < 0.3:
        spatial_low = True

    spatial_high = False
    if isinstance(avg_iou, (int, float)) and isinstance(avg_proximity, (int, float)):
        spatial_high = avg_iou >= 0.5 and avg_proximity >= 0.6
    elif isinstance(avg_iou, (int, float)):
        spatial_high = avg_iou >= 0.5
    elif isinstance(avg_proximity, (int, float)):
        spatial_high = avg_proximity >= 0.6

    if accuracy >= 0.7 and spatial_high:
        quality = "Rendimiento global solido: clasificacion y localizacion estan en zona alta."
    elif accuracy < 0.4 or spatial_low:
        quality = "Rendimiento global debil: conviene revisar prompt, datos y/o modelo."
    else:
        quality = "Rendimiento global intermedio: existe senal util pero aun hay variabilidad."

    metric_status_parts: list[str] = []
    metric_status_parts.append(
        f"IoU medio={float(avg_iou):.3f}" if isinstance(avg_iou, (int, float)) else "IoU medio=N/A"
    )
    metric_status_parts.append(
        (
            f"Proximity medio={float(avg_proximity):.3f}"
            if isinstance(avg_proximity, (int, float))
            else "Proximity medio=N/A"
        )
    )
    metric_status_text = "; ".join(metric_status_parts)

    return (
        f"{quality} Comparativa global: {relation}. "
        f"{metric_status_text}. El mapa de calor muestra en que pares GT-Pred se concentran los aciertos y confusiones, "
        "mientras IoU y Proximity detallan la calidad espacial desde perspectivas complementarias."
    )


def build_metric_charts_reading_guide(*, metric_name: str) -> list[str]:
    """Return user-facing guide lines to interpret metric chart blocks."""
    normalized = _normalize_metric_name(metric_name)

    if normalized == "proximity":
        return [
            "Histograma: muestra como se reparte Proximity en el escenario (cuanto mas cerca de 1, mejor alineacion espacial global).",
            "Barras por clase: compara minimo, media y maximo Proximity en cada clase GT.",
            "Acierto vs fallo: verifica si acertar la clase suele venir acompanado de mejor ajuste espacial global.",
            "Boxplot: resume mediana, dispersion y outliers por clase y por estado de clasificacion.",
            "Curva de umbrales: indica la cobertura acumulada al exigir Proximity minimo.",
        ]

    return [
        "Histograma: muestra como se distribuyen las puntuaciones IoU del escenario.",
        "Barras por clase: compara minimo, media y maximo IoU en cada clase GT.",
        "Acierto vs fallo: contrasta el IoU medio cuando la clase se acierta o se falla.",
        "Boxplot: resume mediana, dispersion y atipicos por clase y estado de clasificacion.",
        "Curva de umbrales: indica la cobertura acumulada al exigir IoU minimo.",
    ]


def build_metric_section_explanation(*, metric_name: str, average_value: float | None) -> str:
    """Build a compact interpretation paragraph for IoU/Proximity sections."""
    normalized = _normalize_metric_name(metric_name)
    metric_label = "Proximity" if normalized == "proximity" else "IoU"

    if average_value is None:
        return (
            f"No hay suficiente informacion para interpretar {metric_label} de forma global en esta ejecucion. "
            "Se recomienda revisar si todas las muestras incluyen bbox GT y bbox predicha validas."
        )

    value = float(average_value)
    if value >= 0.7:
        quality = "alto"
    elif value >= 0.4:
        quality = "intermedio"
    else:
        quality = "bajo"

    if normalized == "proximity":
        return (
            f"El Proximity medio es {value:.4f}, en rango {quality}. "
            "Esta metrica combina cercania de centros y similitud de tamano, por lo que ayuda a detectar ajustes espaciales "
            "razonables incluso cuando IoU no es alto."
        )

    return (
        f"El IoU medio es {value:.4f}, en rango {quality}. "
        "IoU evalua solapamiento estricto entre cajas GT y predicha: valores altos implican localizacion precisa."
    )


def build_classwise_heatmap_explanations(*, labels: list[str], matrix: list[list[int]]) -> list[str]:
    """Generate class-by-class explanation lines from a confusion matrix."""
    if not labels or not matrix:
        return ["No hay suficientes datos para explicar el mapa de calor por clase."]

    explanations: list[str] = []
    for row_index, gt_label in enumerate(labels):
        row = matrix[row_index] if row_index < len(matrix) else []
        total_gt = sum(int(value) for value in row)
        if total_gt <= 0:
            explanations.append(
                f"Clase {gt_label}: no hay casos suficientes en este bloque, asi que no se puede extraer una conclusion fiable."
            )
            continue

        diagonal = int(row[row_index]) if row_index < len(row) else 0
        recall = diagonal / total_gt if total_gt > 0 else 0.0

        top_pred_index = max(range(len(row)), key=lambda idx: int(row[idx])) if row else row_index
        top_pred_label = labels[top_pred_index] if top_pred_index < len(labels) else gt_label
        top_pred_count = int(row[top_pred_index]) if top_pred_index < len(row) else 0

        confusion_parts: list[str] = []
        for col_index, value in enumerate(row):
            if col_index == row_index:
                continue
            count = int(value)
            if count > 0 and col_index < len(labels):
                confusion_parts.append(f"{labels[col_index]} ({count})")
        confusion_text = ", ".join(confusion_parts) if confusion_parts else "no se observan confusiones relevantes"

        approx_ok = int(round(recall * 10.0))
        if recall >= 0.75:
            performance_text = "el comportamiento es solido y bastante consistente"
        elif recall >= 0.5:
            performance_text = "el rendimiento es aceptable, aunque todavia hay margen de mejora"
        elif recall >= 0.3:
            performance_text = "el rendimiento es irregular y aparecen errores frecuentes"
        else:
            performance_text = "el rendimiento es bajo y esta clase necesita atencion prioritaria"

        if top_pred_label == gt_label:
            dominant_text = (
                f"La prediccion mas habitual coincide con la clase real ({top_pred_count} de {total_gt} casos)."
            )
        else:
            dominant_text = (
                f"La confusion dominante lleva esta clase hacia {top_pred_label} "
                f"({top_pred_count} de {total_gt} casos)."
            )

        explanations.append(
            f"Clase {gt_label}: {dominant_text} En terminos sencillos, el modelo acierta aproximadamente "
            f"{approx_ok} de cada 10 casos de esta clase ({recall:.2f}). Globalmente, {performance_text}. "
            f"Confusiones destacadas: {confusion_text}."
        )

    return explanations


def build_top_confusion_errors(
    *,
    labels: list[str],
    matrix: list[list[int]],
    limit: int = 5,
) -> list[str]:
    """Build top off-diagonal confusion errors as readable lines (e.g., HP->AD)."""
    confusions: list[tuple[int, str, str]] = []

    for row_index, gt_label in enumerate(labels):
        if row_index >= len(matrix):
            continue
        row = matrix[row_index]
        for col_index, pred_label in enumerate(labels):
            if row_index == col_index or col_index >= len(row):
                continue
            count = int(row[col_index])
            if count > 0:
                confusions.append((count, gt_label, pred_label))

    confusions.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not confusions:
        return ["No se detectaron confusiones fuera de la diagonal principal."]

    lead_phrases = [
        "Es la confusion mas importante observada",
        "Aparece con frecuencia alta en esta ejecucion",
        "Se repite de forma notable y puede impactar en la lectura clinica",
        "Aunque es menor que las anteriores, sigue siendo relevante",
        "Conviene monitorizar este patron en proximas iteraciones",
    ]

    lines: list[str] = []
    for rank, (count, gt_label, pred_label) in enumerate(confusions[: max(1, int(limit))], start=1):
        gt_index = labels.index(gt_label)
        gt_total = sum(int(value) for value in matrix[gt_index]) if gt_index < len(matrix) else 0
        error_ratio = (count / gt_total * 100.0) if gt_total > 0 else None
        phrase = lead_phrases[min(rank - 1, len(lead_phrases) - 1)]

        if error_ratio is None:
            lines.append(
                f"{rank}) {gt_label}->{pred_label}: {count} casos. {phrase}."
            )
            continue

        if error_ratio >= 50.0:
            severity = "La proporcion es muy alta dentro de esta clase"
        elif error_ratio >= 30.0:
            severity = "La proporcion es alta y merece una revision especifica"
        elif error_ratio >= 15.0:
            severity = "La proporcion es moderada y puede estar afectando al rendimiento global"
        else:
            severity = "La proporcion es baja, pero su seguimiento sigue siendo util"

        lines.append(
            f"{rank}) {gt_label}->{pred_label}: {count} casos, equivalente al {error_ratio:.1f}% de los ejemplos GT={gt_label}. "
            f"{phrase}. {severity}."
        )
    return lines


def _describe_iou_band(iou_value: float | None) -> str:
    """Return a user-friendly qualitative IoU interpretation."""
    if iou_value is None:
        return "No se pudo estimar IoU para este caso, por lo que la calidad espacial es incierta."
    if iou_value >= 0.7:
        return "La superposicion entre GT y prediccion es alta, indicando una localizacion espacial consistente."
    if iou_value >= 0.4:
        return "La superposicion es intermedia: la deteccion localiza bien la zona general, pero aun hay desviacion."
    if iou_value >= 0.2:
        return "La superposicion es baja, con diferencias relevantes entre la caja real y la predicha."
    return "La superposicion es muy baja: la localizacion espacial necesita mejoras claras."


def _describe_proximity_band(proximity_value: float | None) -> str:
    """Return a user-friendly qualitative Proximity interpretation."""
    if proximity_value is None:
        return "No se pudo estimar Proximity para este caso, por lo que la cercania espacial global es incierta."
    if proximity_value >= 0.75:
        return "La cercania espacial global es alta (centro y tamano bien alineados)."
    if proximity_value >= 0.5:
        return "La cercania espacial es intermedia: hay alineacion general, pero con desviaciones apreciables."
    if proximity_value >= 0.3:
        return "La cercania espacial es baja, con diferencias relevantes en posicion y/o tamano."
    return "La cercania espacial es muy baja: la caja predicha queda lejos del objetivo esperado."


def describe_result_item(
    *,
    class_match: bool | None,
    iou_value: float | None,
    proximity_value: float | None = None,
) -> str:
    """Build a concise, easy-to-read explanation for one sample result."""
    if class_match is True:
        class_text = "La clase predicha coincide con la clase real"
    elif class_match is False:
        class_text = "La clase predicha no coincide con la clase real"
    else:
        class_text = "No se pudo confirmar coincidencia de clase"

    iou_text = _describe_iou_band(iou_value)
    proximity_text = _describe_proximity_band(proximity_value)
    return f"{class_text}. {iou_text} {proximity_text}"


def build_executive_summary_text(
    *,
    total_images: int,
    class_accuracy: float | None,
    macro_f1: float | None,
    avg_iou: float | None,
    avg_proximity: float | None = None,
) -> str:
    """Create a high-level narrative summary for non-technical readers."""
    accuracy_text = (
        f"{class_accuracy * 100:.2f}%" if isinstance(class_accuracy, (int, float)) else "N/A"
    )
    macro_f1_text = f"{macro_f1:.4f}" if isinstance(macro_f1, (int, float)) else "N/A"
    iou_text = f"{avg_iou:.4f}" if isinstance(avg_iou, (int, float)) else "N/A"
    proximity_text = f"{avg_proximity:.4f}" if isinstance(avg_proximity, (int, float)) else "N/A"

    return (
        f"Este informe resume {total_images} casos evaluados en el escenario. "
        f"A nivel global, la clasificacion alcanza una exactitud del {accuracy_text}, "
        f"con un Macro-F1 de {macro_f1_text} (equilibrio entre precision y recall por clase) "
        f"y un IoU medio de {iou_text} junto con Proximity medio de {proximity_text} para la calidad de localizacion. "
        "Las secciones siguientes explican de forma guiada donde acierta mas el sistema, "
        "donde confunde clases y que patrones conviene priorizar en iteraciones futuras."
    )
