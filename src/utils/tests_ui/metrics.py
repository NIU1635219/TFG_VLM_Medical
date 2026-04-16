"""Calculo de metricas de evaluacion espacial para Bounding Boxes."""

from __future__ import annotations

import math
from typing import Any, Sequence


def to_float_or_none(value: Any) -> float | None:
    """Convierte a float solo cuando el valor es numerico escalar o cadena numerica.

    - Ignora booleanos (devuelve None).
    - Convierte int/float directamente.
    - Si llega una cadena, intenta convertirla a float (strip(), '' -> None).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def mean_or_none(values: Sequence[float | int]) -> float | None:
    """Devuelve la media de una secuencia numerica o None si esta vacia."""
    if not values:
        return None
    numeric_values = [float(value) for value in values]
    return sum(numeric_values) / len(numeric_values)


def summarize_numeric(values: Sequence[float | int]) -> dict[str, float | None]:
    """Resume una secuencia numerica con media, minimo y maximo."""
    if not values:
        return {"avg": None, "min": None, "max": None}

    numeric_values = [float(value) for value in values]
    avg_value = mean_or_none(numeric_values)
    return {
        "avg": avg_value,
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def _validate_box(box: list[int], box_name: str) -> None:
    """Valida la estructura y la consistencia geometrica de una bbox normalizada."""
    if len(box) != 4:
        raise ValueError(f"{box_name} debe tener exactamente 4 elementos [xmin, ymin, xmax, ymax].")

    if any(not isinstance(value, int) for value in box):
        raise ValueError(f"{box_name} debe contener solo enteros.")

    xmin, ymin, xmax, ymax = box
    if xmin > xmax:
        raise ValueError(f"{box_name} es invalida: xmin no puede ser mayor que xmax.")
    if ymin > ymax:
        raise ValueError(f"{box_name} es invalida: ymin no puede ser mayor que ymax.")


def calculate_iou(boxA: list[int], boxB: list[int]) -> float:
    """
    Calcula la metrica IoU (Intersection over Union) entre dos Bounding Boxes.

    Esta funcion recibe dos cajas en formato normalizado estilo Qwen Visual Grounding:
    [xmin, ymin, xmax, ymax], donde cada coordenada es un entero en el rango 0-1000.

    IoU se define como:
    area_interseccion / area_union

    donde:
    - area_interseccion es el area comun entre ambas cajas.
    - area_union es la suma de areas individuales menos la interseccion.

    Comportamiento esperado:
    - Devuelve un float entre 0.0 y 1.0.
    - Si las cajas no se solapan, devuelve 0.0.
    - Si alguna caja es invalida (por ejemplo ymin > ymax o xmin > xmax),
      lanza ValueError.

    Args:
        boxA: Primera bbox en formato [xmin, ymin, xmax, ymax].
        boxB: Segunda bbox en formato [xmin, ymin, xmax, ymax].

    Returns:
        Valor IoU en el rango [0.0, 1.0].
    """
    _validate_box(boxA, "boxA")
    _validate_box(boxB, "boxB")

    x_inter_min = max(boxA[0], boxB[0])
    y_inter_min = max(boxA[1], boxB[1])
    x_inter_max = min(boxA[2], boxB[2])
    y_inter_max = min(boxA[3], boxB[3])

    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_height * inter_width

    area_a = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_b = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_center_distance_score(boxA: list[int], boxB: list[int], *, alpha: float = 3.0) -> float:
    """Calcula score [0,1] según distancia entre centros normalizada por diagonal conjunta."""
    _validate_box(boxA, "boxA")
    _validate_box(boxB, "boxB")

    center_a_x = (boxA[0] + boxA[2]) / 2.0
    center_a_y = (boxA[1] + boxA[3]) / 2.0
    center_b_x = (boxB[0] + boxB[2]) / 2.0
    center_b_y = (boxB[1] + boxB[3]) / 2.0

    delta_y = center_a_y - center_b_y
    delta_x = center_a_x - center_b_x
    center_distance = math.sqrt((delta_y * delta_y) + (delta_x * delta_x))

    x_min = min(boxA[0], boxB[0])
    y_min = min(boxA[1], boxB[1])
    x_max = max(boxA[2], boxB[2])
    y_max = max(boxA[3], boxB[3])
    envelope_w = max(1.0, float(x_max - x_min))
    envelope_h = max(1.0, float(y_max - y_min))
    envelope_diag = math.sqrt((envelope_h * envelope_h) + (envelope_w * envelope_w))

    norm_dist = center_distance / envelope_diag
    score = math.exp(-float(alpha) * (norm_dist * norm_dist))
    return max(0.0, min(1.0, float(score)))


def calculate_size_relative_score(boxA: list[int], boxB: list[int]) -> float:
    """Calcula score [0,1] por diferencia relativa de área entre dos cajas."""
    _validate_box(boxA, "boxA")
    _validate_box(boxB, "boxB")

    area_a = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_b = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = float(max(1, area_a, area_b))
    rel_error = abs(float(area_a) - float(area_b)) / denom
    score = 1.0 - rel_error
    return max(0.0, min(1.0, float(score)))


def _intersection_area(boxA: list[int], boxB: list[int]) -> int:
    """Devuelve area de interseccion (0 si no hay solapamiento)."""
    x_inter_min = max(boxA[0], boxB[0])
    y_inter_min = max(boxA[1], boxB[1])
    x_inter_max = min(boxA[2], boxB[2])
    y_inter_max = min(boxA[3], boxB[3])

    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    return inter_height * inter_width


def _boxes_touch_or_overlap(boxA: list[int], boxB: list[int]) -> bool:
    """True cuando las cajas se tocan o se solapan (sin hueco entre ellas)."""
    separated_x = boxA[2] < boxB[0] or boxB[2] < boxA[0]
    separated_y = boxA[3] < boxB[1] or boxB[3] < boxA[1]
    return not (separated_y or separated_x)


def _containment_ratio(boxA: list[int], boxB: list[int]) -> float:
    """Mide contencion relativa: interseccion / area de la caja mas pequena."""
    inter_area = _intersection_area(boxA, boxB)
    area_a = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_b = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    return max(0.0, min(1.0, float(inter_area) / float(min_area)))


def calculate_proximity_score(
    boxA: list[int],
    boxB: list[int],
    *,
    center_weight: float = 0.6,
    size_weight: float = 0.4,
    alpha: float = 3.0,
) -> dict[str, float]:
    """Calcula score combinado de proximidad y sus componentes en [0,1].

    Criterio geométrico reforzado:
    - Si las cajas no se tocan, la proximidad final se degrada a casi cero.
    - Si hay contacto pero poca contención, la proximidad también se penaliza.
    - La contención alta (una caja dentro de otra) conserva puntuaciones altas.
    """
    _validate_box(boxA, "boxA")
    _validate_box(boxB, "boxB")

    center_score = calculate_center_distance_score(boxA, boxB, alpha=alpha)
    size_score = calculate_size_relative_score(boxA, boxB)

    c_weight = float(center_weight)
    s_weight = float(size_weight)
    weight_sum = c_weight + s_weight
    if weight_sum <= 0:
        c_weight, s_weight = 0.6, 0.4
        weight_sum = 1.0

    combined = ((c_weight * center_score) + (s_weight * size_score)) / weight_sum

    touch_or_overlap = _boxes_touch_or_overlap(boxA, boxB)
    contain_ratio = _containment_ratio(boxA, boxB)

    # Si no hay contacto espacial, no se puede considerar una localizacion buena.
    if not touch_or_overlap:
        combined *= 0.02
    else:
        # Penaliza parcialidad y favorece casos de contencion clara.
        overlap_gate = 0.1 + (0.9 * contain_ratio)
        combined *= overlap_gate

    return {
        "proximity_score": max(0.0, min(1.0, float(combined))),
        "proximity_center_score": center_score,
        "proximity_size_score": size_score,
    }
