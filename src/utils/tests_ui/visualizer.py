from __future__ import annotations

from pathlib import Path
from typing import Sequence
from typing import Final

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


SCALE_MAX: Final[int] = 1000


def _load_image_bgr(image_path: Path) -> np.ndarray:
    """Carga imagen en BGR usando OpenCV y fallback a PIL si es necesario."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is not None:
        return image_bgr

    if Image is not None:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    raise RuntimeError(f"No se pudo cargar la imagen: {image_path}")


def _validate_bbox(bbox: list[int], bbox_name: str) -> None:
    """Valida que el bounding box tenga el formato esperado [ymin, xmin, ymax, xmax]."""
    if len(bbox) != 4:
        raise ValueError(f"{bbox_name} debe tener exactamente 4 elementos: [ymin, xmin, ymax, xmax].")

    if any(not isinstance(value, int) for value in bbox):
        raise TypeError(f"{bbox_name} debe contener solo enteros.")


def _denormalize_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    """Convierte un bbox normalizado (0-1000) a coordenadas en píxeles de la imagen."""
    ymin_norm, xmin_norm, ymax_norm, xmax_norm = bbox

    xmin = int((xmin_norm / SCALE_MAX) * width)
    ymin = int((ymin_norm / SCALE_MAX) * height)
    xmax = int((xmax_norm / SCALE_MAX) * width)
    ymax = int((ymax_norm / SCALE_MAX) * height)

    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width - 1))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height - 1))

    return xmin, ymin, xmax, ymax


def draw_comparison_bboxes(
    image_path: str,
    gt_bbox: list[int] | None,
    pred_bbox: list[int],
    model_name: str,
    iou_score: float | None = None,
    gt_color: tuple[int, int, int] | None = None,
    pred_color: tuple[int, int, int] | None = None,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Dibuja el bbox de Ground Truth y el bbox predicho sobre la imagen original.

    Args:
        image_path: Ruta de la imagen de entrada.
        gt_bbox: Bounding box Ground Truth normalizado [ymin, xmin, ymax, xmax] en escala 0-1000.
            Si es None, se omite el dibujo de Ground Truth.
        pred_bbox: Bounding box predicho normalizado [ymin, xmin, ymax, xmax] en escala 0-1000.
        model_name: Nombre del modelo que generó la predicción.
        iou_score: Valor de IoU opcional para mostrar en el texto superpuesto.
        output_path: Ruta opcional de guardado; si no se proporciona, la imagen se muestra por pantalla.

    Returns:
        np.ndarray: Imagen en formato RGB con los bounding boxes dibujados.

    Raises:
        FileNotFoundError: Si la imagen de entrada no existe.
        ValueError: Si el bbox no tiene formato válido.
        RuntimeError: Si OpenCV no puede cargar la imagen.
    """
    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {source_path}")

    if gt_bbox is not None:
        _validate_bbox(gt_bbox, "gt_bbox")
    _validate_bbox(pred_bbox, "pred_bbox")

    image_bgr = _load_image_bgr(source_path)

    height, width = image_bgr.shape[:2]

    gt_pixel_bbox: tuple[int, int, int, int] | None = None
    if gt_bbox is not None:
        gt_pixel_bbox = _denormalize_bbox(gt_bbox, width, height)
    pr_xmin, pr_ymin, pr_xmax, pr_ymax = _denormalize_bbox(pred_bbox, width, height)

    # Colores por defecto: verde para GT, rojo para predicción.
    gt_col = gt_color if gt_color is not None else (0, 255, 0)
    pred_col = pred_color if pred_color is not None else (0, 0, 255)

    if gt_pixel_bbox is not None:
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_pixel_bbox
        cv2.rectangle(image_bgr, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), gt_col, 2)

    cv2.rectangle(image_bgr, (pr_xmin, pr_ymin), (pr_xmax, pr_ymax), pred_col, 2)

    overlay_text = model_name.strip() or "Model"
    if iou_score is not None:
        overlay_text = f"{overlay_text} | IoU: {iou_score:.2f}"

    text_org = (10, 30)
    cv2.putText(
        image_bgr,
        overlay_text,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        overlay_text,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(destination), image_bgr)
        if not success:
            raise RuntimeError(f"No se pudo guardar la imagen en: {destination}")
    else:
        plt.figure(figsize=(10, 7))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title("Comparacion de Bounding Boxes")
        plt.show()

    return image_rgb


def draw_multi_comparison_bboxes(
    image_path: str,
    gt_bboxes: Sequence[list[int] | None],
    pred_bboxes: Sequence[list[int]],
    model_name: str,
    iou_scores: Sequence[float | None] | None = None,
    gt_color: tuple[int, int, int] | None = None,
    pred_color: tuple[int, int, int] | None = None,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Dibuja multiples bounding boxes GT/pred sobre una unica imagen.

    Args:
        image_path: Ruta de la imagen de entrada.
        gt_bboxes: Lista de bboxes GT (puede contener None en elementos sin referencia).
        pred_bboxes: Lista de bboxes predichas en formato [ymin, xmin, ymax, xmax].
        model_name: Nombre del modelo.
        iou_scores: Lista opcional de IoU por deteccion.
        output_path: Ruta opcional de guardado.

    Returns:
        Imagen RGB con todas las cajas dibujadas.
    """
    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {source_path}")

    if len(pred_bboxes) == 0:
        raise ValueError("pred_bboxes no puede estar vacio.")

    if len(gt_bboxes) != len(pred_bboxes):
        raise ValueError("gt_bboxes y pred_bboxes deben tener la misma longitud.")

    if iou_scores is not None and len(iou_scores) != len(pred_bboxes):
        raise ValueError("iou_scores debe tener la misma longitud que pred_bboxes.")

    image_bgr = _load_image_bgr(source_path)

    height, width = image_bgr.shape[:2]

    # Colores por defecto: verde para GT, rojo para predicción.
    gt_col = gt_color if gt_color is not None else (0, 255, 0)
    pred_col = pred_color if pred_color is not None else (0, 0, 255)

    for idx, pred_bbox in enumerate(pred_bboxes, start=1):
        _validate_bbox(pred_bbox, f"pred_bbox_{idx}")
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = _denormalize_bbox(pred_bbox, width, height)
        cv2.rectangle(image_bgr, (pred_xmin, pred_ymin), (pred_xmax, pred_ymax), pred_col, 2)

        gt_bbox = gt_bboxes[idx - 1]
        if gt_bbox is not None:
            _validate_bbox(gt_bbox, f"gt_bbox_{idx}")
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = _denormalize_bbox(gt_bbox, width, height)
            cv2.rectangle(image_bgr, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), gt_col, 2)

        score_text = ""
        if iou_scores is not None and iou_scores[idx - 1] is not None:
            score_text = f" IoU:{iou_scores[idx - 1]:.2f}"

        label_pos = (max(0, pred_xmin), max(20, pred_ymin - 6))
        label = f"#{idx}{score_text}"
        cv2.putText(image_bgr, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    overlay_text = model_name.strip() or "Model"
    if iou_scores:
        valid_scores = [score for score in iou_scores if score is not None]
        if valid_scores:
            mean_iou = sum(valid_scores) / len(valid_scores)
            overlay_text = f"{overlay_text} | mean IoU: {mean_iou:.2f}"

    text_org = (10, 30)
    cv2.putText(image_bgr, overlay_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, overlay_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(destination), image_bgr)
        if not success:
            raise RuntimeError(f"No se pudo guardar la imagen en: {destination}")
    else:
        plt.figure(figsize=(10, 7))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title("Comparacion de Multiples Bounding Boxes")
        plt.show()

    return image_rgb


def draw_predicted_bboxes(
    image_path: str,
    pred_bboxes: Sequence[list[int]],
    labels: Sequence[str] | None = None,
    pred_color: tuple[int, int, int] | None = None,
    output_path: str | None = None,
) -> np.ndarray:
    """Dibuja multiples bboxes de prediccion sobre una imagen."""
    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {source_path}")

    if len(pred_bboxes) == 0:
        raise ValueError("pred_bboxes no puede estar vacio.")

    if labels is not None and len(labels) != len(pred_bboxes):
        raise ValueError("labels debe tener la misma longitud que pred_bboxes.")

    image_bgr = _load_image_bgr(source_path)
    height, width = image_bgr.shape[:2]

    pred_col = pred_color if pred_color is not None else (0, 0, 255)

    for idx, pred_bbox in enumerate(pred_bboxes, start=1):
        _validate_bbox(pred_bbox, f"pred_bbox_{idx}")
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = _denormalize_bbox(pred_bbox, width, height)
        cv2.rectangle(image_bgr, (pred_xmin, pred_ymin), (pred_xmax, pred_ymax), pred_col, 2)

        label_text = labels[idx - 1] if labels is not None else ""
        caption = f"#{idx}" if not label_text else f"#{idx} {label_text}"
        label_pos = (max(0, pred_xmin), max(20, pred_ymin - 6))
        cv2.putText(image_bgr, caption, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, caption, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(destination), image_bgr)
        if not success:
            raise RuntimeError(f"No se pudo guardar la imagen en: {destination}")
    else:
        plt.figure(figsize=(10, 7))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title("Predicciones")
        plt.show()

    return image_rgb
