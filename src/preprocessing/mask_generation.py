from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def generate_mask_from_image(image_bgr: np.ndarray) -> np.ndarray:
    """Genera una máscara heurística de pólipo a partir de una imagen BGR.

    Esta función no sustituye a un segmentador entrenado. Se usa como respaldo
    para visualización y como baseline de segmentación simple cuando no existe GT.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold_mask.mean() < 127.0:
        threshold_mask = cv2.bitwise_not(threshold_mask)

    kernel = np.ones((5, 5), dtype=np.uint8)
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contour_result: Any = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        pseudo_mask = np.zeros_like(threshold_mask)
        cv2.drawContours(pseudo_mask, [largest], -1, 255, thickness=-1)
        return (pseudo_mask > 0).astype(np.uint8)

    return (threshold_mask > 0).astype(np.uint8)


# Alias más explícito para el notebook.
infer_pseudo_mask = generate_mask_from_image
