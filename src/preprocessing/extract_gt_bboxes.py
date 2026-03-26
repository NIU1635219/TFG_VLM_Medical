from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Extrae bounding boxes desde máscaras binarias (.tif/.tiff), "
            "normaliza en escala 0-1000 y exporta el resultado a CSV."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directorio raíz del dataset de entrada.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/ground_truth_bboxes.csv"),
        help="Ruta del CSV de salida.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe el CSV de salida si ya existe.",
    )
    return parser


def find_mask_files(input_dir: Path) -> list[Path]:
    """Localiza recursivamente todas las máscaras .tif/.tiff dentro de carpetas masks."""
    mask_paths: set[Path] = set()

    for pattern in ("**/masks/**/*.tif", "**/masks/**/*.tiff"):
        for mask_path in input_dir.rglob(pattern):
            if mask_path.is_file():
                mask_paths.add(mask_path)

    return sorted(mask_paths)


def extract_normalized_bbox(mask_path: Path) -> list[int]:
    """
    Extrae y normaliza un bounding box a escala 0-1000 desde una máscara.

    Devuelve el bbox en formato [ymin, xmin, ymax, xmax]. Si no hay contornos,
    devuelve [0, 0, 0, 0].
    """
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [0, 0, 0, 0]

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [0, 0, 0, 0]

    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    height, width = image.shape[:2]
    xmin = int((x / width) * 1000)
    ymin = int((y / height) * 1000)
    xmax = int(((x + w) / width) * 1000)
    ymax = int(((y + h) / height) * 1000)

    return [ymin, xmin, ymax, xmax]


def build_bbox_table(mask_paths: list[Path]) -> pd.DataFrame:
    """Genera una tabla con image_id y bounding box normalizado para cada máscara."""
    records: list[dict[str, str | int]] = []

    for mask_path in tqdm(mask_paths, desc="Extrayendo BBoxes", unit="mask"):
        ymin, xmin, ymax, xmax = extract_normalized_bbox(mask_path)
        records.append(
            {
                "image_id": mask_path.stem,
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                "mask_path": str(mask_path),
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> int:
    """Punto de entrada principal del script."""
    args = build_parser().parse_args()

    input_dir: Path = args.input_dir
    output_csv: Path = args.output_csv
    force: bool = args.force

    if output_csv.exists() and not force:
        print(f"El archivo de salida ya existe: {output_csv}. Usa --force para sobrescribir.")
        return 0

    if not input_dir.exists():
        print(f"[ERROR] No existe el directorio de entrada: {input_dir}")
        return 1

    mask_paths = find_mask_files(input_dir)
    if not mask_paths:
        print(f"[WARN] No se encontraron máscaras .tif/.tiff en carpetas 'masks' dentro de: {input_dir}")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe = build_bbox_table(mask_paths)
    dataframe.to_csv(output_csv, index=False)

    print("\n=== Resumen de extracción de Ground Truth BBoxes ===")
    print(f"Entrada: {input_dir}")
    print(f"Total máscaras procesadas: {len(mask_paths)}")
    print(f"CSV generado: {output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
