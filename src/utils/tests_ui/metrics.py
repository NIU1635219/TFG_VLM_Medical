"""Calculo de metricas de evaluacion espacial para Bounding Boxes."""

from __future__ import annotations

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
        raise ValueError(f"{box_name} debe tener exactamente 4 elementos [ymin, xmin, ymax, xmax].")

    if any(not isinstance(value, int) for value in box):
        raise ValueError(f"{box_name} debe contener solo enteros.")

    ymin, xmin, ymax, xmax = box
    if ymin > ymax:
        raise ValueError(f"{box_name} es invalida: ymin no puede ser mayor que ymax.")
    if xmin > xmax:
        raise ValueError(f"{box_name} es invalida: xmin no puede ser mayor que xmax.")


def calculate_iou(boxA: list[int], boxB: list[int]) -> float:
    """
    Calcula la metrica IoU (Intersection over Union) entre dos Bounding Boxes.

    Esta funcion recibe dos cajas en formato normalizado estilo Detectron/PyTorch:
    [ymin, xmin, ymax, xmax], donde cada coordenada es un entero en el rango 0-1000.

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
        boxA: Primera bbox en formato [ymin, xmin, ymax, xmax].
        boxB: Segunda bbox en formato [ymin, xmin, ymax, xmax].

    Returns:
        Valor IoU en el rango [0.0, 1.0].
    """
    _validate_box(boxA, "boxA")
    _validate_box(boxB, "boxB")

    y_inter_min = max(boxA[0], boxB[0])
    x_inter_min = max(boxA[1], boxB[1])
    y_inter_max = min(boxA[2], boxB[2])
    x_inter_max = min(boxA[3], boxB[3])

    inter_height = max(0, y_inter_max - y_inter_min)
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_area = inter_height * inter_width

    area_a = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_b = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area
