from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


VALID_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


@dataclass
class CropResult:
    """
    Clase de datos que representa el resultado de una operación de recorte.

    Attributes:
        input_path (Path): Ruta del archivo de imagen original.
        output_path (Path): Ruta donde se guardó la imagen procesada.
        was_cropped (bool): Indica si la imagen fue recortada o no.
        bbox (tuple[int, int, int, int]): Cuadro delimitador aplicado (x, y, w, h).
    """
    input_path: Path
    output_path: Path
    was_cropped: bool
    bbox: tuple[int, int, int, int]


def find_endoscopy_bbox(image_bgr, min_area_ratio: float = 0.1) -> tuple[int, int, int, int]:
    """
    Encuentra el cuadro delimitador (bbox) de la región útil en una imagen de endoscopia.

    Intenta eliminar los bordes negros típicos de estas imágenes mediante umbralización
    y detección de contornos.

    Args:
        image_bgr (numpy.ndarray): Imagen de entrada en formato BGR.
        min_area_ratio (float, optional): Ratio mínimo del área del contorno respecto
            al tamaño total de la imagen para ser considerado válido. Por defecto es 0.1.

    Returns:
        tuple[int, int, int, int]: Una tupla (x, y, w, h) representando el bbox.
            Si no se encuentra un contorno válido, devuelve el tamaño completo de la imagen.
    """
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, width, height)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    min_area = int(width * height * min_area_ratio)
    if w * h < min_area:
        return (0, 0, width, height)

    return (x, y, w, h)


def crop_image_remove_black_borders(image_bgr, min_area_ratio: float = 0.1):
    """
    Recorta los bordes negros de una imagen de endoscopia.

    Utiliza `find_endoscopy_bbox` para determinar el área de interés y realiza el recorte.

    Args:
        image_bgr (numpy.ndarray): Imagen de entrada en formato BGR.
        min_area_ratio (float, optional): Ratio mínimo de área para considerar válido el recorte.

    Returns:
        tuple: Una tupla conteniendo:
            - cropped (numpy.ndarray): La imagen recortada (o la original si no se recortó).
            - bbox (tuple): El cuadro delimitador usado (x, y, w, h).
    """
    x, y, w, h = find_endoscopy_bbox(image_bgr, min_area_ratio=min_area_ratio)
    cropped = image_bgr[y : y + h, x : x + w]
    return cropped, (x, y, w, h)


def iter_images(root_dir: Path):
    """
    Itera recursivamente sobre un directorio devolviendo rutas a archivos de imagen válidos.

    Args:
        root_dir (Path): Directorio raíz donde buscar imágenes.

    Yields:
        Path: Ruta a cada archivo de imagen encontrado.
    """
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def iter_csv_files(root_dir: Path):
    """
    Itera recursivamente sobre un directorio devolviendo rutas a archivos CSV.

    Args:
        root_dir (Path): Directorio raíz donde buscar archivos CSV.

    Yields:
        Path: Ruta a cada archivo CSV encontrado.
    """
    for path in sorted(root_dir.rglob("*.csv")):
        if path.is_file():
            yield path


def copy_csv_files(input_dir: Path, output_dir: Path, dry_run: bool = False) -> int:
    """
    Copia todos los archivos CSV desde el directorio de entrada al de salida, manteniendo la estructura.

    Args:
        input_dir (Path): Directorio de origen.
        output_dir (Path): Directorio de destino.
        dry_run (bool, optional): Si es True, no realiza la copia física. Por defecto es False.

    Returns:
        int: El número de archivos copiados.
    """
    csv_paths = list(iter_csv_files(input_dir))
    copied_count = 0

    for csv_path in csv_paths:
        relative_path = csv_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(csv_path.read_bytes())

        copied_count += 1

    return copied_count


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    max_images: int | None = None,
    min_area_ratio: float = 0.1,
    dry_run: bool = False,
) -> list[CropResult]:
    """
    Procesa un dataset completo de imágenes recortando bordes negros.

    Itera sobre las imágenes en el directorio de entrada, aplica el recorte y
    guarda los resultados en el directorio de salida manteniendo la estructura.

    Args:
        input_dir (Path): Directorio raíz del dataset original.
        output_dir (Path): Directorio donde guardar el dataset procesado.
        max_images (int, optional): Límite máximo de imágenes a procesar.
        min_area_ratio (float, optional): Ratio mínimo de área para detección de bbox.
        dry_run (bool, optional): Si es True, simula el proceso sin escribir archivos.

    Returns:
        list[CropResult]: Lista de resultados con detalles de cada imagen procesada.
    """
    image_paths = list(iter_images(input_dir))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    results: list[CropResult] = []

    for image_path in image_paths:
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] No se pudo leer: {image_path}")
            continue

        cropped, bbox = crop_image_remove_black_borders(image, min_area_ratio=min_area_ratio)
        was_cropped = cropped.shape[:2] != image.shape[:2]

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cropped)

        results.append(
            CropResult(
                input_path=image_path,
                output_path=output_path,
                was_cropped=was_cropped,
                bbox=bbox,
            )
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesa imágenes endoscópicas: detecta el contorno principal del campo "
            "visible y recorta bordes negros."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directorio de entrada con imágenes originales.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directorio de salida para imágenes procesadas.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Límite de imágenes a procesar (útil para pruebas rápidas).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.1,
        help="Área mínima relativa del contorno para aceptar recorte (0-1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula recortes sin escribir archivos en disco.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"[ERROR] No existe el directorio de entrada: {input_dir}")
        return 1

    results = process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        max_images=args.max_images,
        min_area_ratio=args.min_area_ratio,
        dry_run=args.dry_run,
    )
    csv_copied_count = copy_csv_files(
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )

    total = len(results)
    cropped_count = sum(1 for result in results if result.was_cropped)

    print("\n=== Resumen de preprocesado ===")
    print(f"Entrada: {input_dir}")
    print(f"Salida: {output_dir}")
    print(f"Imágenes procesadas: {total}")
    print(f"Imágenes recortadas: {cropped_count}")
    print(f"Imágenes sin cambios: {total - cropped_count}")
    print(f"CSV duplicados: {csv_copied_count}")
    print(f"Modo dry-run: {'sí' if args.dry_run else 'no'}")

    if total == 0:
        print("[WARN] No se encontraron imágenes compatibles para procesar.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
