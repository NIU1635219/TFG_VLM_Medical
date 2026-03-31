from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
from tqdm import tqdm

DEFAULT_SPLITS: tuple[str, ...] = ("m_train", "m_valid")
EMPTY_BBOX_COLUMNS: tuple[str, ...] = (
    "image_id",
    "ymin",
    "xmin",
    "ymax",
    "xmax",
    "mask_path",
)


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
        default=Path("data/processed/m_train/ground_truth_bboxes.csv"),
        help="Ruta del CSV de salida (modo simple).",
    )
    parser.add_argument(
        "--split-aware",
        action="store_true",
        help=(
            "Genera un CSV por split m_* dentro de --input-dir "
            "(por defecto: m_train/m_valid)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Raíz de salida para modo --split-aware. Si se omite, usa --input-dir. "
            "Ejemplo: --input-dir data/raw --output-root data/raw"
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Lista de splits a procesar en modo --split-aware (por defecto: m_train m_valid).",
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


def _normalize_splits(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        split = str(value).strip()
        if not split:
            continue
        if split in seen:
            continue
        seen.add(split)
        normalized.append(split)
    return normalized


def _empty_bbox_table() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EMPTY_BBOX_COLUMNS))


def _export_one(input_dir: Path, output_csv: Path, *, force: bool) -> tuple[int, str]:
    if output_csv.exists() and not force:
        return 0, f"[SKIP] Ya existe: {output_csv}. Usa --force para sobrescribir."

    if not input_dir.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        _empty_bbox_table().to_csv(output_csv, index=False)
        return 0, (
            f"[WARN] No existe el directorio de entrada: {input_dir}. "
            f"Se genera CSV vacío en {output_csv}."
        )

    mask_paths = find_mask_files(input_dir)
    if not mask_paths:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        _empty_bbox_table().to_csv(output_csv, index=False)
        return 0, (
            "[WARN] No se encontraron máscaras .tif/.tiff en carpetas 'masks' "
            f"dentro de: {input_dir}. Se genera CSV vacío en {output_csv}."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe = build_bbox_table(mask_paths)
    dataframe.to_csv(output_csv, index=False)

    summary = "\n".join(
        [
            "=== Resumen de extracción de Ground Truth BBoxes ===",
            f"Entrada: {input_dir}",
            f"Total máscaras procesadas: {len(mask_paths)}",
            f"CSV generado: {output_csv}",
        ]
    )
    return 0, summary


def main() -> int:
    """Punto de entrada principal del script."""
    args = build_parser().parse_args()

    input_dir: Path = args.input_dir
    output_csv: Path = args.output_csv
    output_root: Path | None = args.output_root
    force: bool = args.force
    split_aware: bool = bool(args.split_aware)
    splits = _normalize_splits(args.splits or DEFAULT_SPLITS)

    if split_aware:
        if not input_dir.exists():
            print(f"[ERROR] No existe el directorio de entrada: {input_dir}")
            return 1

        effective_output_root = output_root if output_root is not None else input_dir
        if not splits:
            print("[ERROR] Debes indicar al menos un split en --splits para --split-aware.")
            return 1

        print("=== Extracción por split activada ===")
        print(f"Entrada base: {input_dir}")
        print(f"Salida base: {effective_output_root}")
        print(f"Splits: {', '.join(splits)}")

        failures = 0
        generated: list[Path] = []
        for split in splits:
            split_input = input_dir / split
            split_output = effective_output_root / split / "ground_truth_bboxes.csv"
            print(f"\n--- Procesando split: {split} ---")
            code, message = _export_one(split_input, split_output, force=force)
            print(message)
            if code == 0 and split_output.exists():
                generated.append(split_output)
            else:
                failures += 1

        if generated:
            print("\n=== CSVs por split generados ===")
            for csv_path in generated:
                print(f"- {csv_path}")

        if failures > 0:
            print(f"\n[ERROR] Fallaron {failures} splits.")
            return 1
        return 0

    code, message = _export_one(input_dir, output_csv, force=force)
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
