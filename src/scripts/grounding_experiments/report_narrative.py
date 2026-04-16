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


def _quality_band(value: float | None, *, low: float, high: float) -> str:
    if not isinstance(value, (int, float)):
        return "desconocido"
    numeric = float(value)
    if numeric < low:
        return "bajo"
    if numeric < high:
        return "intermedio"
    return "alto"


def _coverage_text(available: int, total: int) -> str:
    if total <= 0:
        return "0/0 (sin muestras)"
    ratio = (available / total) * 100.0
    return f"{available}/{total} ({ratio:.1f}%)"


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


def build_classification_quality_explanation(
    *,
    class_accuracy: float | None,
    macro_f1: float | None,
    recall_by_class: dict[str, float] | None = None,
) -> str:
    """Explain classification quality using complementary metrics."""
    if class_accuracy is None and macro_f1 is None:
        return (
            "No hay metrica suficiente para interpretar la calidad de clasificacion. "
            "Se necesitan etiquetas GT y predicciones validas."
        )

    acc_band = _quality_band(class_accuracy, low=0.45, high=0.7)
    f1_band = _quality_band(macro_f1, low=0.4, high=0.65)

    if class_accuracy is not None and macro_f1 is not None:
        if class_accuracy >= 0.7 and macro_f1 >= 0.65:
            base = (
                "La clasificacion es solida tanto en exactitud global como en equilibrio entre clases, "
                "lo que reduce el riesgo de sobreoptimizar solo la clase mayoritaria."
            )
        elif class_accuracy >= 0.65 and macro_f1 < 0.45:
            base = (
                "La exactitud global es aceptable, pero el Macro-F1 es bajo; esto sugiere desbalance en errores "
                "y posible castigo sobre clases minoritarias."
            )
        elif class_accuracy < 0.45:
            base = (
                "La exactitud global es baja y la capacidad de discriminacion por clase necesita mejora prioritaria."
            )
        else:
            base = (
                "La clasificacion es intermedia: hay señal util, pero aun existen confusiones persistentes entre clases."
            )
    elif class_accuracy is not None:
        base = (
            f"La exactitud global es {acc_band}. "
            "Sin Macro-F1 no puede evaluarse con precision el balance entre clases."
        )
    else:
        base = (
            f"El Macro-F1 es {f1_band}. "
            "Sin accuracy global no puede estimarse el porcentaje total de aciertos."
        )

    if not isinstance(recall_by_class, dict) or not recall_by_class:
        return base

    recalls = [float(v) for v in recall_by_class.values() if isinstance(v, (int, float))]
    if not recalls:
        return base

    min_recall = min(recalls)
    max_recall = max(recalls)
    spread = max_recall - min_recall
    spread_comment: str
    if spread >= 0.4:
        spread_comment = (
            "La variabilidad de recall entre clases es alta, por lo que conviene atacar confusiones especificas por clase."
        )
    elif spread >= 0.2:
        spread_comment = (
            "La variabilidad de recall es moderada y recomienda ajustes focalizados en las clases con menor cobertura."
        )
    else:
        spread_comment = "La variabilidad de recall es contenida y el comportamiento entre clases es relativamente uniforme."

    return f"{base} {spread_comment}"


def build_spatial_consistency_explanation(
    *,
    avg_iou: float | None,
    avg_proximity: float | None,
) -> str:
    """Explain when IoU and Proximity agree or diverge."""
    if avg_iou is None and avg_proximity is None:
        return "No hay metricas espaciales suficientes para interpretar consistencia geometrica."
    if avg_iou is None:
        return (
            "Solo hay Proximity medio disponible. Esto permite valorar cercania centro-tamano, "
            "pero no el solapamiento estricto de cajas (IoU)."
        )
    if avg_proximity is None:
        return (
            "Solo hay IoU medio disponible. Esto mide solapamiento estricto, "
            "pero no separa explicitamente alineacion de centro y escala."
        )

    iou = float(avg_iou)
    prox = float(avg_proximity)
    gap = prox - iou

    if gap >= 0.2:
        relation = (
            "Proximity supera claramente a IoU: el modelo suele aproximar posicion y escala, "
            "pero falla en ajustar con precision los bordes."
        )
    elif gap <= -0.1:
        relation = (
            "IoU no queda por debajo de Proximity: puede haber buen solapamiento puntual, "
            "aunque no siempre estable en centro/tamano a nivel global."
        )
    else:
        relation = "IoU y Proximity son coherentes entre si, indicando una calidad espacial global consistente."

    iou_band = _quality_band(iou, low=0.35, high=0.6)
    prox_band = _quality_band(prox, low=0.45, high=0.7)
    return (
        f"Calidad espacial: IoU en rango {iou_band} y Proximity en rango {prox_band}. "
        f"{relation}"
    )


def build_timing_performance_explanation(
    *,
    total_images: int,
    total_duration_count: int,
    avg_total_duration_seconds: float | None,
    total_inference_seconds: float | None,
    avg_generation_seconds: float | None,
    avg_ttft_seconds: float | None,
    avg_tokens_per_second: float | None,
) -> str:
    """Explain timing quality and telemetry completeness."""
    if total_images <= 0:
        return "No hay muestras procesadas para analizar latencia global."

    coverage = _coverage_text(total_duration_count, total_images)
    if avg_total_duration_seconds is None and avg_generation_seconds is None and avg_ttft_seconds is None:
        return (
            f"Cobertura temporal: {coverage}. No hay valores suficientes de latencia para una lectura temporal fiable."
        )

    latency_text = (
        f"latencia total media={float(avg_total_duration_seconds):.3f} s"
        if isinstance(avg_total_duration_seconds, (int, float))
        else "latencia total media=N/A"
    )
    total_text = (
        f"latencia total acumulada={float(total_inference_seconds):.3f} s"
        if isinstance(total_inference_seconds, (int, float))
        else "latencia total acumulada=N/A"
    )
    gen_text = (
        f"generacion media={float(avg_generation_seconds):.3f} s"
        if isinstance(avg_generation_seconds, (int, float))
        else "generacion media=N/A"
    )
    ttft_text = (
        f"TTFT medio={float(avg_ttft_seconds):.3f} s"
        if isinstance(avg_ttft_seconds, (int, float))
        else "TTFT medio=N/A"
    )
    tps_text = (
        f"TPS medio={float(avg_tokens_per_second):.2f} tok/s"
        if isinstance(avg_tokens_per_second, (int, float))
        else "TPS medio=N/A"
    )

    if isinstance(avg_total_duration_seconds, (int, float)):
        latency = float(avg_total_duration_seconds)
        if latency <= 1.0:
            latency_band = "baja"
        elif latency <= 3.0:
            latency_band = "moderada"
        else:
            latency_band = "alta"
        latency_note = f"La latencia media se considera {latency_band} para este flujo."
    else:
        latency_note = "No hay latencia total media para clasificar el rendimiento temporal."

    return (
        f"Cobertura temporal: {coverage}. {latency_text}; {total_text}; {gen_text}; {ttft_text}; {tps_text}. "
        f"{latency_note}"
    )


def build_metrics_completeness_explanation(
    *,
    total_images: int,
    iou_count: int,
    proximity_count: int,
    class_match_count: int,
    total_duration_count: int,
) -> str:
    """Explain data completeness for each reported metric group."""
    if total_images <= 0:
        return "No hay registros para evaluar cobertura de metricas."

    iou_cov = _coverage_text(iou_count, total_images)
    prox_cov = _coverage_text(proximity_count, total_images)
    cls_cov = _coverage_text(class_match_count, total_images)
    time_cov = _coverage_text(total_duration_count, total_images)

    return (
        "Cobertura de metricas del informe: "
        f"clasificacion={cls_cov}, IoU={iou_cov}, Proximity={prox_cov}, tiempos={time_cov}. "
        "Si alguna cobertura es baja, la interpretacion asociada debe leerse con cautela."
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


def _describe_timing_band(
    *,
    total_duration_seconds: float | None,
    generation_duration_seconds: float | None,
    ttft_seconds: float | None,
) -> str:
    """Return a compact timing interpretation for one inference."""
    if (
        total_duration_seconds is None
        and generation_duration_seconds is None
        and ttft_seconds is None
    ):
        return "No hay telemetria temporal suficiente para evaluar la latencia de esta inferencia."

    timing_parts: list[str] = []
    if isinstance(total_duration_seconds, (int, float)):
        total_value = float(total_duration_seconds)
        if total_value <= 1.0:
            timing_parts.append(f"La latencia total es baja ({total_value:.3f} s)")
        elif total_value <= 3.0:
            timing_parts.append(f"La latencia total es moderada ({total_value:.3f} s)")
        else:
            timing_parts.append(f"La latencia total es alta ({total_value:.3f} s)")

    if isinstance(generation_duration_seconds, (int, float)):
        generation_value = float(generation_duration_seconds)
        timing_parts.append(f"el tiempo de generacion es {generation_value:.3f} s")

    if isinstance(ttft_seconds, (int, float)):
        ttft_value = float(ttft_seconds)
        timing_parts.append(f"y el TTFT es {ttft_value:.3f} s")

    if not timing_parts:
        return "Hay datos temporales parciales, pero no suficientes para una lectura clara de latencia."

    return ", ".join(timing_parts) + "."


def describe_result_item(
    *,
    class_match: bool | None,
    iou_value: float | None,
    proximity_value: float | None = None,
    total_duration_seconds: float | None = None,
    generation_duration_seconds: float | None = None,
    ttft_seconds: float | None = None,
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
    timing_text = _describe_timing_band(
        total_duration_seconds=total_duration_seconds,
        generation_duration_seconds=generation_duration_seconds,
        ttft_seconds=ttft_seconds,
    )

    recommendation: str
    if class_match is True and isinstance(iou_value, (int, float)) and iou_value >= 0.5:
        recommendation = "Caso favorable: conviene usarlo como referencia positiva para calibracion futura."
    elif class_match is False and isinstance(iou_value, (int, float)) and iou_value < 0.3:
        recommendation = (
            "Caso critico: revisar prompt de clasificacion y estrategia de localizacion, "
            "porque fallan clase y ajuste espacial a la vez."
        )
    elif class_match is False:
        recommendation = "La localizacion puede ser util, pero la clase requiere ajuste para reducir confusiones."
    else:
        recommendation = "La clase es correcta, aunque aun hay margen para mejorar la precision espacial."

    return f"{class_text}. {iou_text} {proximity_text} {timing_text} {recommendation}"


def build_case_followup_recommendation(
    *,
    class_match: bool | None,
    iou_value: float | None,
    proximity_value: float | None,
    total_duration_seconds: float | None = None,
) -> str:
    """Return a specific, actionable follow-up suggestion for one sample."""
    if class_match is False and isinstance(iou_value, (int, float)) and iou_value < 0.2:
        return (
            "Prioridad alta: revisar este caso en detalle (prompt + parsing + visualizacion), "
            "porque combina error de clase con localizacion muy deficiente."
        )

    if class_match is True and isinstance(iou_value, (int, float)) and iou_value >= 0.6:
        if isinstance(total_duration_seconds, (int, float)) and float(total_duration_seconds) > 3.0:
            return (
                "Caso de referencia en calidad, pero con coste temporal alto; "
                "puede usarse para explorar optimizaciones de latencia sin perder precision."
            )
        return (
            "Caso de referencia positiva: conviene conservarlo como ejemplo valido para regresion y comparativas de modelo."
        )

    if class_match is True and isinstance(proximity_value, (int, float)) and proximity_value < 0.35:
        return (
            "La clase es correcta pero el ajuste espacial es flojo; "
            "recomendable priorizar refinamiento de bbox o instrucciones de localizacion."
        )

    if class_match is False and isinstance(proximity_value, (int, float)) and proximity_value >= 0.6:
        return (
            "La geometria es razonable pero la clase falla; "
            "centrar mejoras en desambiguacion semantica entre AD/HP/ASS."
        )

    return (
        "Caso mixto: mantener seguimiento en proximas iteraciones y contrastar con patrones similares del reporte."
    )


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
