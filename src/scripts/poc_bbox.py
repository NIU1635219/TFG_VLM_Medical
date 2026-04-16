"""
PoC de Visual Grounding con Bounding Boxes.
Descarga automáticamente imágenes de prueba si no existen y valida la localización local.
Utiliza un System Prompt avanzado para mejorar la precisión del modelo.
"""

import os
import sys
import argparse
import time
import json
from typing import Any, Callable
import urllib.request
import urllib.error
from pathlib import Path
import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

# Permitir imports de src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.schemas import BoundingBox
from src.inference.vlm_runner import VLMLoader
from src.utils.tests_ui.metrics import calculate_iou, mean_or_none, to_float_or_none
from src.utils.tests_ui.markdown_report import (
    ImageGroupSection,
    ImageItem,
    ListSection,
    SectionGroup,
    write_markdown_report,
)
from src.utils.tests_ui.visualizer import draw_multi_comparison_bboxes, draw_predicted_bboxes

# --- CONFIGURACIÓN ---
TEST_IMAGE_DIR = Path("data/smoke_test/bbox_test")
RESULTS_DIR = Path("data/processed/bbox_results")

# --- PROMPT INTELIGENTE (Visual Grounding Expert) ---
SYSTEM_PROMPT = (
    "You are an expert visual grounding AI. Locate the requested subject in the image. "
    "First reason about how many relevant subjects are present, then provide the total detected subject count. "
    "Return zero, one, or many detections; for each detection include detected_subject and bounding box "
    "coordinates normalized to 0-1000 as xmin, ymin, xmax, ymax. "
    "detected_subject must be a descriptive phrase (not a single word), including visual cues "
    "such as color/pose/type and rough location in the image. "
    "If none are present, return an empty detections list and count 0."
)

# --- URLs DE MUESTRAS ACTUALIZADAS ---
SAMPLES = {
    "cat_1.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=1000&auto=format&fit=crop",
    "cat_2.jpg": "https://images.unsplash.com/photo-1548247416-ec66f4900b2e?auto=format&fit=crop&q=80&w=1000",
    "cat_3.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    "cat_4.jpg": "https://purina.com.pa/sites/default/files/2025-09/Conoce-las-razas-de-gatos.jpg",
    "cat_5.jpg": "https://cdn.sanity.io/images/5vm5yn1d/pro/41c7e7ce298604b0801fc2b1b76371a47e9ebb83-950x633.jpg?fm=webp&q=80",
    "cat_6.jpg": "https://upload.wikimedia.org/wikipedia/commons/6/64/Collage_of_Six_Cats-02.jpg",
    "dog_1.jpg": "https://images.unsplash.com/photo-1517849845537-4d257902454a?q=80&w=1000&auto=format&fit=crop",
    "dog_2.jpg": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?q=80&w=1000&auto=format&fit=crop",
    "dog_3.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=1000&auto=format&fit=crop",
    "dog_4.jpg": "https://images.pexels.com/photos/850602/pexels-photo-850602.jpeg?165=2817",
    "dog_5.jpg": "https://thumbs.dreamstime.com/b/vista-delantera-de-dos-perros-del-boxeador-sent%C3%A1ndose-12911045.jpg",
    "dog_6.jpg": "https://img.freepik.com/foto-gratis/coleccion-retratos-cachorros-adorables_53876-145628.jpg?semt=ais_hybrid&w=740&q=80",
}

Reporter = Callable[[str, dict[str, Any]], None]
SCALE_MAX = 1000

# Ground Truth manual para evaluación visual y cálculo de IoU en la PoC.
GT_BBOXES: dict[str, list[dict[str, Any]]] = {
    "cat_1.jpg": [
        {"detected_subject": "Gato bicolor", "bbox": [84, 166, 964, 909]},
    ],
    "cat_2.jpg": [
        {"detected_subject": "Gato gris", "bbox": [175, 297, 970, 983]},
    ],
    "cat_3.jpg": [
        {"detected_subject": "Gato naranja", "bbox": [1, 1, 1000, 785]},
    ],
    "cat_4.jpg": [
        {"detected_subject": "Gatito tricolor (izquierda)", "bbox": [212, 201, 867, 444]},
        {"detected_subject": "Gatito atigrado (centro izquierda)", "bbox": [209, 424, 855, 542]},
        {"detected_subject": "Gatito naranja (centro derecha)", "bbox": [220, 488, 863, 633]},
        {"detected_subject": "Gatito crema (derecha)", "bbox": [297, 617, 851, 763]},
    ],
    "cat_5.jpg": [
        {"detected_subject": "Gato naranja", "bbox": [177, 330, 935, 638]},
        {"detected_subject": "Gato atigrado", "bbox": [90, 473, 949, 898]},
    ],
    "cat_6.jpg": [
        {"detected_subject": "Gato negro (arriba izquierda)", "bbox": [14, 37, 475, 266]},
        {"detected_subject": "Gato blanco y naranja (arriba centro)", "bbox": [62, 418, 471, 631]},
        {"detected_subject": "Gato gris pelo largo (arriba derecha)", "bbox": [21, 718, 477, 938]},
        {"detected_subject": "Gato siamés (abajo izquierda)", "bbox": [552, 78, 953, 317]},
        {"detected_subject": "Gato blanco y negro (abajo centro)", "bbox": [539, 346, 986, 636]},
        {"detected_subject": "Gato tricolor (abajo derecha)", "bbox": [525, 693, 1000, 990]},
    ],
    "dog_1.jpg": [
        {"detected_subject": "Perro pug negro", "bbox": [360, 177, 1000, 1000]},
    ],
    "dog_2.jpg": [
        {"detected_subject": "Perro bulldog francés amarillo", "bbox": [257, 344, 907, 597]},
    ],
    "dog_3.jpg": [
        {"detected_subject": "Perro beagle", "bbox": [129, 189, 1000, 750]},
    ],
    "dog_4.jpg": [
        {"detected_subject": "Perro atigrado (izquierda)", "bbox": [182, 77, 1000, 404]},
        {"detected_subject": "Perro blanco (centro)", "bbox": [227, 368, 1000, 705]},
        {"detected_subject": "Perro marrón (derecha)", "bbox": [84, 675, 1000, 967]},
    ],
    "dog_5.jpg": [
        {"detected_subject": "Perro boxer marrón (izquierda)", "bbox": [81, 157, 976, 519]},
        {"detected_subject": "Perro boxer atigrado (derecha)", "bbox": [68, 492, 975, 883]},
    ],
    "dog_6.jpg": [
        {"detected_subject": "Perro orejas caídas (arriba izquierda)", "bbox": [105, 50, 501, 282]},
        {"detected_subject": "Pug negro (arriba centro)", "bbox": [138, 379, 501, 604]},
        {"detected_subject": "Perro orejas levantadas (arriba derecha)", "bbox": [113, 683, 499, 985]},
        {"detected_subject": "Perro marrón (abajo izquierda)", "bbox": [605, 42, 1000, 292]},
        {"detected_subject": "Bulldog (abajo centro)", "bbox": [578, 350, 1000, 648]},
        {"detected_subject": "Cachorro crema (abajo derecha)", "bbox": [585, 705, 1000, 964]},
    ],
}


def _default_reporter(event: str, payload: dict[str, Any]) -> None:
    """Reporter por defecto para ejecución CLI (stdout)."""
    if event == "download_start":
        print(f"📥 Descargando: {payload['name']}...")
    elif event == "download_retry":
        print(f"⚠️ Servidor saturado. Reintentando en {payload['wait_seconds']}s...")
    elif event == "download_ok":
        print("   ✅ Listo.")
    elif event == "download_error":
        print(f"❌ Imposible descargar {payload['name']}: {payload['error']}")
    elif event == "blank_generated":
        print("⬜ Imagen en blanco generada.")
    elif event == "run_start":
        print(f"\n🚀 Iniciando PoC: {payload['model_tag']}\n")
    elif event == "image_start":
        print(f"🔍 Analizando {payload['image_name']}...")
    elif event == "image_result":
        print(f"   ✅ Subjects detectados: {payload['count']}")
        detections = payload.get("detections") or []
        if not detections:
            print("   ℹ️ Sin detecciones para esta imagen.")
        for idx, detection in enumerate(detections, start=1):
            print(
                "   📍 "
                f"#{idx} {detection['detected_subject']}: "
                f"[{detection['ymin']}, {detection['xmin']}, {detection['ymax']}, {detection['xmax']}]"
            )
            if all(key in detection for key in ("px_ymin", "px_xmin", "px_ymax", "px_xmax")):
                print(
                    "      🧭 px: "
                    f"[{detection['px_ymin']}, {detection['px_xmin']}, "
                    f"{detection['px_ymax']}, {detection['px_xmax']}]"
                )
    elif event == "image_error":
        print(f"   ❌ Error: {payload['error']}")
    elif event == "run_saved":
        print(f"\n✨ Resultados guardados en:\n📂 {payload['output_dir']}")
        if payload.get("markdown_path"):
            print(f"📝 Reporte Markdown: {payload['markdown_path']}")


def _emit(reporter: Reporter | None, event: str, **payload: Any) -> None:
    if reporter is not None:
        reporter(event, payload)


def _download_with_headers(url: str, dest_path: Path, *, reporter: Reporter | None = None):
    """Descarga con reintentos y cabeceras de navegador para evitar Error 403."""
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        }
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(dest_path, 'wb') as out_file:
                    out_file.write(response.read())
                return
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 5
                _emit(reporter, "download_retry", wait_seconds=wait)
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError(f"No se pudo descargar tras {max_retries} intentos.")


def ensure_assets(*, reporter: Reporter | None = None):
    """Descarga imágenes solo si faltan, con control de flujo."""
    TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, url in SAMPLES.items():
        path = TEST_IMAGE_DIR / name
        if not path.exists():
            _emit(reporter, "download_start", name=name)
            try:
                _download_with_headers(url, path, reporter=reporter)
                _emit(reporter, "download_ok", name=name)
                time.sleep(1) # Pausa ligera entre descargas
            except Exception as e:
                _emit(reporter, "download_error", name=name, error=str(e))

    # Imagen en blanco para control negativo (validar que no alucina)
    blank_path = TEST_IMAGE_DIR / "blank_sample.jpg"
    if not blank_path.exists():
        cv2.imwrite(str(blank_path), np.full((768, 1024, 3), 255, dtype=np.uint8))
        _emit(reporter, "blank_generated")

def _load_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Obtiene dimensiones reales (alto, ancho) de una imagen."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        return h, w

    if Image is not None:
        try:
            with Image.open(image_path) as pil_img:
                w, h = pil_img.size
            return h, w
        except Exception:
            pass

    # Fallback defensivo para no abortar la ejecución completa si el archivo no es decodificable.
    return 1000, 1000


def _denormalize_bbox_to_pixels(
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Convierte coordenadas normalizadas 0-1000 a píxeles y las acota a la imagen."""
    px_xmin = int((xmin / SCALE_MAX) * width)
    px_ymin = int((ymin / SCALE_MAX) * height)
    px_xmax = int((xmax / SCALE_MAX) * width)
    px_ymax = int((ymax / SCALE_MAX) * height)

    px_xmin = max(0, min(px_xmin, width - 1))
    px_xmax = max(0, min(px_xmax, width - 1))
    px_ymin = max(0, min(px_ymin, height - 1))
    px_ymax = max(0, min(px_ymax, height - 1))

    return px_ymin, px_xmin, px_ymax, px_xmax


def _save_run_results(results_dir: Path, model_tag: str, records: list[dict[str, Any]]):
    """Guarda el JSONL y las imágenes en una carpeta con timestamp."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"run_{run_id}"
    annotated_dir = run_dir / "annotated"
    comparison_dir = run_dir / "comparison"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = run_dir / "results.jsonl"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            # Procesar imagen anotada
            if rec["status"] == "ok":
                try:
                    pred_bboxes = [
                        [det.xmin, det.ymin, det.xmax, det.ymax]
                        for det in rec["bbox_obj"].detections
                    ]
                    pred_labels = [
                        f"{rec['subject']} | {det.detected_subject}"
                        for det in rec["bbox_obj"].detections
                    ]
                    img_annotated = draw_predicted_bboxes(
                        image_path=str(rec["image_path"]),
                        pred_bboxes=pred_bboxes,
                        labels=pred_labels,
                    )
                    out_name = f"res_{Path(rec['image_path']).name}"
                    # draw_predicted_bboxes returns an RGB image for display. When
                    # saving with OpenCV we must convert back to BGR to avoid
                    # channel-swapping (red <-> blue) in the written file.
                    try:
                        img_to_save = cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR)
                    except Exception:
                        img_to_save = img_annotated
                    cv2.imwrite(str(annotated_dir / out_name), img_to_save)
                    rec["annotated_path"] = str(annotated_dir / out_name)

                except Exception as error:
                    rec["annotated_error"] = str(error)

                try:

                    comparison_metrics: list[dict[str, Any]] = []
                    gt_items = GT_BBOXES.get(Path(rec["image_path"]).name, [])
                    pred_bboxes: list[list[int]] = []
                    gt_bboxes: list[list[int] | None] = []
                    iou_scores: list[float | None] = []
                    for idx, det in enumerate(rec["bbox_obj"].detections, start=1):
                        pred_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                        gt_item = gt_items[idx - 1] if idx - 1 < len(gt_items) else None
                        gt_bbox = gt_item["bbox"] if gt_item is not None else None
                        iou_score = calculate_iou(pred_bbox, gt_bbox) if gt_bbox is not None else None
                        pred_bboxes.append(pred_bbox)
                        gt_bboxes.append(gt_bbox)
                        iou_scores.append(iou_score)
                        comparison_metrics.append(
                            {
                                "detection_index": idx,
                                "pred_bbox": pred_bbox,
                                "gt_bbox": gt_bbox,
                                "gt_subject": gt_item["detected_subject"] if gt_item is not None else None,
                                "iou": iou_score,
                            }
                        )

                    if pred_bboxes:
                        comparison_name = f"cmp_{Path(rec['image_path']).stem}.jpg"
                        comparison_path = comparison_dir / comparison_name
                        draw_multi_comparison_bboxes(
                            image_path=str(rec["image_path"]),
                            gt_bboxes=gt_bboxes,
                            pred_bboxes=pred_bboxes,
                            model_name=model_tag,
                            iou_scores=iou_scores,
                            output_path=str(comparison_path),
                        )
                        rec["comparison_path"] = str(comparison_path)
                        for metric in comparison_metrics:
                            metric["comparison_path"] = str(comparison_path)

                    if comparison_metrics:
                        rec["comparison_metrics"] = comparison_metrics
                except Exception as error:
                    rec["comparison_error"] = str(error)
            
            # Limpiar objeto Pydantic para serializar JSON
            data = rec.copy()
            data.pop("bbox_obj", None)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    all_ious: list[float] = []
    for rec in records:
        for metric in rec.get("comparison_metrics") or []:
            iou_value = metric.get("iou") if isinstance(metric, dict) else None
            iou_float = to_float_or_none(iou_value)
            if iou_float is not None:
                all_ious.append(iou_float)

    global_iou_avg: float | None = mean_or_none(all_ious)

    report_sections: list[SectionGroup] = [
        SectionGroup(
            heading="Leyenda",
            heading_level=2,
            sections=[
                ListSection(
                    items=[
                        "GT: verde",
                        "Prediccion del modelo: rojo",
                    ],
                    ordered=False,
                    heading_level=3,
                )
            ],
        )
    ]
    for idx, rec in enumerate(records, start=1):
        base_info = [
            f"Imagen: {Path(rec['image_path']).name}",
            f"Subject prompt: {rec.get('subject', '-')}",
            f"Estado: {rec.get('status', '-')}",
        ]

        if rec.get("status") == "ok":
            base_info.append(f"Detected subjects count: {rec.get('detected_subjects_count', 0)}")
            base_info.append(f"Resolucion: {rec.get('image_width', '-') }x{rec.get('image_height', '-')} px")
            rec_ious: list[float] = []
            for metric in (rec.get("comparison_metrics") or []):
                if not isinstance(metric, dict):
                    continue
                iou_float = to_float_or_none(metric.get("iou"))
                if iou_float is not None:
                    rec_ious.append(iou_float)
            rec_iou_avg: float | None = mean_or_none(rec_ious)
            base_info.append(
                f"IoU medio: {rec_iou_avg:.3f}" if rec_iou_avg is not None else "IoU medio: N/A"
            )
            if rec.get("object_count_reasoning"):
                base_info.append(f"Reasoning: {rec['object_count_reasoning']}")
        elif rec.get("error"):
            base_info.append(f"Error: {rec['error']}")

        item_sections: list[Any] = [
            ListSection(items=base_info, heading="Resumen", ordered=False, heading_level=3),
        ]

        detections = rec.get("detections") or []
        if detections:
            iou_by_detection: dict[int, float | None] = {}
            for metric in rec.get("comparison_metrics") or []:
                if not isinstance(metric, dict):
                    continue
                det_index = metric.get("detection_index")
                if isinstance(det_index, int):
                    iou_by_detection[det_index] = to_float_or_none(metric.get("iou"))

            det_lines: list[str] = []
            for det_idx, det in enumerate(detections, start=1):
                iou_value = iou_by_detection.get(det_idx)
                det_lines.append(
                    f"#{det_idx} {det.get('detected_subject', '-')}: "
                    f"IoU={iou_value:.3f}" if iou_value is not None else f"#{det_idx} {det.get('detected_subject', '-')}: IoU=N/A"
                )
            item_sections.append(ListSection(items=det_lines, heading="Detecciones", ordered=False, heading_level=3))

        image_items: list[ImageItem] = [
            ImageItem(path=rec["image_path"], alt_text=f"original_{idx}", caption="Imagen original"),
        ]
        if rec.get("annotated_path"):
            image_items.append(
                ImageItem(path=rec["annotated_path"], alt_text=f"annotated_{idx}", caption="Imagen anotada (predicciones)")
            )
        if rec.get("comparison_path"):
            image_items.append(
                ImageItem(path=rec["comparison_path"], alt_text=f"comparison_{idx}", caption="Comparacion GT vs prediccion")
            )
        if image_items:
            item_sections.append(ImageGroupSection(images=image_items, heading="Visualizaciones", heading_level=3))

        report_sections.append(
            SectionGroup(
                heading=f"Resultado {idx}: {Path(rec['image_path']).name}",
                heading_level=2,
                sections=item_sections,
            )
        )

    markdown_path = run_dir / "results.md"
    write_markdown_report(
        report_path=markdown_path,
        title="PoC BBox Report",
        metadata={
            "Model": model_tag,
            "Run ID": run_id,
            "Total images": str(len(records)),
            "JSONL": str(jsonl_path.name),
            "Global IoU (avg)": f"{global_iou_avg:.3f}" if global_iou_avg is not None else "N/A",
        },
        sections=report_sections,
    )

    return jsonl_path, markdown_path

def main(argv=None, *, reporter: Reporter | None = None):
    if reporter is None:
        reporter = _default_reporter

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ID del modelo")
    parser.add_argument("--host", default=None, help="Host de LM Studio API, p. ej. localhost:1234")
    parser.add_argument("--api-token", default=None, help="Token de API para LM Studio (opcional)")
    args = parser.parse_args(argv)

    ensure_assets(reporter=reporter)
    loader = VLMLoader(
        model_path=args.model,
        verbose=True,
        server_api_host=args.host,
        api_token=args.api_token,
    )
    images = sorted(list(TEST_IMAGE_DIR.glob("*.jpg")))

    run_records = []
    _emit(reporter, "run_start", model_tag=args.model, total_images=len(images))

    for img_path in images:
        subject = "dog" if "dog" in img_path.name else "cat"
        if "blank" in img_path.name: subject = "animal"

        _emit(reporter, "image_start", image_name=img_path.name)
        
        record = {
            "image_path": str(img_path),
            "subject": subject,
            "status": "error",
            "timestamp": time.time()
        }

        try:
            result = loader.inference(
                image_path=str(img_path),
                prompt=f"Locate the {subject} in this image.",
                schema=BoundingBox,
                system_prompt=SYSTEM_PROMPT
            )
            if result.detected_subjects_count != len(result.detections):
                raise ValueError(
                    "Respuesta inconsistente del modelo: "
                    f"detected_subjects_count={result.detected_subjects_count} "
                    f"pero detections={len(result.detections)}"
                )

            image_height, image_width = _load_image_dimensions(img_path)
            detections_payload = []
            for det in result.detections:
                px_ymin, px_xmin, px_ymax, px_xmax = _denormalize_bbox_to_pixels(
                    ymin=det.ymin,
                    xmin=det.xmin,
                    ymax=det.ymax,
                    xmax=det.xmax,
                    height=image_height,
                    width=image_width,
                )
                detections_payload.append(
                    {
                        "detected_subject": det.detected_subject,
                        "ymin": det.ymin,
                        "xmin": det.xmin,
                        "ymax": det.ymax,
                        "xmax": det.xmax,
                        "px_ymin": px_ymin,
                        "px_xmin": px_xmin,
                        "px_ymax": px_ymax,
                        "px_xmax": px_xmax,
                    }
                )
            _emit(
                reporter,
                "image_result",
                image_name=img_path.name,
                count=result.detected_subjects_count,
                object_count_reasoning=result.object_count_reasoning,
                image_height=image_height,
                image_width=image_width,
                detections=detections_payload,
            )
            
            record.update({
                "status": "ok",
                "object_count_reasoning": result.object_count_reasoning,
                "detected_subjects_count": result.detected_subjects_count,
                "image_height": image_height,
                "image_width": image_width,
                "detections": detections_payload,
                "bbox_obj": result # Temporal para dibujo
            })
        except Exception as e:
            _emit(reporter, "image_error", image_name=img_path.name, error=str(e))
            record["error"] = str(e)
        
        run_records.append(record)

    # Guardar todo
    jsonl_file, markdown_file = _save_run_results(RESULTS_DIR, args.model, run_records)
    _emit(
        reporter,
        "run_saved",
        output_dir=str(jsonl_file.parent.absolute()),
        markdown_path=str(markdown_file.absolute()),
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())