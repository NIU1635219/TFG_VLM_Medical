from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkPaths:
    project_root: Path
    raw_root: Path
    train_root: Path
    valid_root: Path
    test_root: Path
    train_csv: Path
    valid_csv: Path
    test_csv: Path
    train_image_dir: Path
    train_mask_dir: Path
    valid_image_dir: Path
    valid_mask_dir: Path
    test_image_dir: Path


class SplitSpec(TypedDict):
    csv: Path
    image_dir: Path
    mask_dir: Optional[Path]


class SplitBundle(TypedDict):
    train: tuple[pd.DataFrame, pd.DataFrame]
    valid: tuple[pd.DataFrame, pd.DataFrame]
    test: tuple[pd.DataFrame, pd.DataFrame]


def find_project_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("No se encontró pyproject.toml")


PROJECT_ROOT = find_project_root()
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
TRAIN_ROOT = RAW_ROOT / "m_train"
VALID_ROOT = RAW_ROOT / "m_valid"
TEST_ROOT = RAW_ROOT / "m_test"
TRAIN_CSV = TRAIN_ROOT / "train.csv"
VALID_CSV = VALID_ROOT / "valid.csv"
TEST_CSV = TEST_ROOT / "gt_test.csv"
TRAIN_IMAGE_DIR = TRAIN_ROOT / "images"
TRAIN_MASK_DIR = TRAIN_ROOT / "masks"
VALID_IMAGE_DIR = VALID_ROOT / "images"
VALID_MASK_DIR = VALID_ROOT / "masks"
TEST_IMAGE_DIR = TEST_ROOT / "images"

PATHS = BenchmarkPaths(
    project_root=PROJECT_ROOT,
    raw_root=RAW_ROOT,
    train_root=TRAIN_ROOT,
    valid_root=VALID_ROOT,
    test_root=TEST_ROOT,
    train_csv=TRAIN_CSV,
    valid_csv=VALID_CSV,
    test_csv=TEST_CSV,
    train_image_dir=TRAIN_IMAGE_DIR,
    train_mask_dir=TRAIN_MASK_DIR,
    valid_image_dir=VALID_IMAGE_DIR,
    valid_mask_dir=VALID_MASK_DIR,
    test_image_dir=TEST_IMAGE_DIR,
)

SPLIT_SPECS: dict[str, SplitSpec] = {
    "train": {"csv": TRAIN_CSV, "image_dir": TRAIN_IMAGE_DIR, "mask_dir": TRAIN_MASK_DIR},
    "valid": {"csv": VALID_CSV, "image_dir": VALID_IMAGE_DIR, "mask_dir": VALID_MASK_DIR},
    "test": {"csv": TEST_CSV, "image_dir": TEST_IMAGE_DIR, "mask_dir": None},
}


def find_by_stem(folder: Path, stem: str) -> Path | None:
    candidates = [path for path in folder.rglob("*") if path.is_file() and path.stem == stem]
    if not candidates:
        return None
    preferred_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    candidates = sorted(candidates, key=lambda path: (path.suffix.lower() not in preferred_suffixes, path.name))
    return candidates[0]


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"No se pudo leer la máscara: {path}")
    return (mask > 0).astype(np.uint8)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = image_bgr.astype(np.float32).copy()
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255.0
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color[mask_bool] * alpha
    return overlay.astype(np.uint8)


def load_split_records(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPLIT_SPECS[split_name]
    raw_rows = pd.read_csv(spec["csv"]).to_dict(orient="records")
    image_dir = spec["image_dir"]
    mask_dir = spec["mask_dir"]
    has_masks = mask_dir is not None

    records: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []

    for row_index, raw_row in enumerate(raw_rows):
        image_id = str(raw_row.get("image_id", "")).strip()
        cls = str(raw_row.get("cls", "")).upper().strip()
        histologia = raw_row.get("Histologia")
        if not image_id or not cls:
            continue

        image_path = find_by_stem(image_dir, image_id)
        mask_path = find_by_stem(mask_dir, image_id) if has_masks and mask_dir is not None else None
        if image_path is None or (has_masks and mask_path is None):
            missing.append(
                {
                    "split": split_name,
                    "image_id": image_id,
                    "image_path": str(image_path) if image_path else None,
                    "mask_path": str(mask_path) if mask_path else None,
                }
            )
            continue

        records.append(
            {
                "split": split_name,
                "row_index": row_index,
                "image_id": image_id,
                "cls": cls,
                "Histologia": histologia,
                "image_path": image_path,
                "mask_path": mask_path,
                "mask_source": "gt" if mask_path is not None else "none",
            }
        )

    return pd.DataFrame(records), pd.DataFrame(missing)


def load_benchmark_splits() -> SplitBundle:
    return {
        "train": load_split_records("train"),
        "valid": load_split_records("valid"),
        "test": load_split_records("test"),
    }