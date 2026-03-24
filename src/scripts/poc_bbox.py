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

from src.inference.schemas import BoundingBox, BoundingBoxDetection
from src.inference.vlm_runner import VLMLoader

# --- CONFIGURACIÓN ---
TEST_IMAGE_DIR = Path("data/smoke_test/bbox_test")
RESULTS_DIR = Path("data/processed/bbox_results")

# --- PROMPT INTELIGENTE (Visual Grounding Expert) ---
SYSTEM_PROMPT = (
    "You are an expert visual grounding AI. Locate the requested subject in the image. "
    "First reason about how many relevant subjects are present, then provide the total detected subject count. "
    "Return zero, one, or many detections; for each detection include detected_subject and bounding box "
    "coordinates normalized to 0-1000 as ymin, xmin, ymax, xmax. "
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
    elif event == "image_error":
        print(f"   ❌ Error: {payload['error']}")
    elif event == "run_saved":
        print(f"\n✨ Resultados guardados en:\n📂 {payload['output_dir']}")


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

def _draw_bboxes(image_path: Path, detections: list[BoundingBoxDetection], label: str) -> np.ndarray:
    """Carga imagen y dibuja cero, una o múltiples cajas escaladas."""
    img = cv2.imread(str(image_path))
    if img is None and Image is not None:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None

    if img is None:
        raise RuntimeError(f"Error leyendo {image_path}")
    
    h, w = img.shape[:2]
    for idx, bbox in enumerate(detections, start=1):
        x1 = int(bbox.xmin * w / 1000)
        y1 = int(bbox.ymin * h / 1000)
        x2 = int(bbox.xmax * w / 1000)
        y2 = int(bbox.ymax * h / 1000)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        caption = f"{label} #{idx} | {bbox.detected_subject}"
        cv2.putText(
            img,
            caption,
            (max(0, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return img

def _save_run_results(results_dir: Path, model_tag: str, records: list[dict[str, Any]]):
    """Guarda el JSONL y las imágenes en una carpeta con timestamp."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"run_{run_id}"
    annotated_dir = run_dir / "annotated"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = run_dir / "results.jsonl"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            # Procesar imagen anotada
            if rec["status"] == "ok":
                try:
                    img_annotated = _draw_bboxes(
                        Path(rec["image_path"]),
                        rec["bbox_obj"].detections,
                        rec["subject"],
                    )
                    out_name = f"res_{Path(rec['image_path']).name}"
                    cv2.imwrite(str(annotated_dir / out_name), img_annotated)
                    rec["annotated_path"] = str(annotated_dir / out_name)
                except Exception as error:
                    rec["annotated_error"] = str(error)
            
            # Limpiar objeto Pydantic para serializar JSON
            data = rec.copy()
            data.pop("bbox_obj", None)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    return jsonl_path

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

            detections_payload = [
                {
                    "detected_subject": det.detected_subject,
                    "ymin": det.ymin,
                    "xmin": det.xmin,
                    "ymax": det.ymax,
                    "xmax": det.xmax,
                }
                for det in result.detections
            ]
            _emit(
                reporter,
                "image_result",
                image_name=img_path.name,
                count=result.detected_subjects_count,
                object_count_reasoning=result.object_count_reasoning,
                detections=detections_payload,
            )
            
            record.update({
                "status": "ok",
                "object_count_reasoning": result.object_count_reasoning,
                "detected_subjects_count": result.detected_subjects_count,
                "detections": detections_payload,
                "bbox_obj": result # Temporal para dibujo
            })
        except Exception as e:
            _emit(reporter, "image_error", image_name=img_path.name, error=str(e))
            record["error"] = str(e)
        
        run_records.append(record)

    # Guardar todo
    jsonl_file = _save_run_results(RESULTS_DIR, args.model, run_records)
    _emit(reporter, "run_saved", output_dir=str(jsonl_file.parent.absolute()))
    return 0

if __name__ == "__main__":
    sys.exit(main())