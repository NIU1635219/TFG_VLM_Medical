from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TypedDict, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing.polyp_benchmark_utils import (
    PATHS as COMMON_PATHS,
    PROJECT_ROOT as COMMON_PROJECT_ROOT,
    RAW_ROOT as COMMON_RAW_ROOT,
    SPLIT_SPECS as COMMON_SPLIT_SPECS,
    TEST_CSV as COMMON_TEST_CSV,
    TEST_IMAGE_DIR as COMMON_TEST_IMAGE_DIR,
    TEST_ROOT as COMMON_TEST_ROOT,
    TRAIN_CSV as COMMON_TRAIN_CSV,
    TRAIN_IMAGE_DIR as COMMON_TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR as COMMON_TRAIN_MASK_DIR,
    TRAIN_ROOT as COMMON_TRAIN_ROOT,
    VALID_CSV as COMMON_VALID_CSV,
    VALID_IMAGE_DIR as COMMON_VALID_IMAGE_DIR,
    VALID_MASK_DIR as COMMON_VALID_MASK_DIR,
    VALID_ROOT as COMMON_VALID_ROOT,
    find_by_stem as common_find_by_stem,
    find_project_root as common_find_project_root,
    load_image_bgr as common_load_image_bgr,
    load_mask as common_load_mask,
    load_split_records as common_load_split_records,
    overlay_mask as common_overlay_mask,
)

try:
    import importlib

    skimage_hog = cast(Any, importlib.import_module("skimage.feature")).hog
except Exception:
    skimage_hog = None


RANDOM_STATE = 42
CLASS_ORDER = ["AD", "HP", "ASS"]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class SplitSpec(TypedDict):
    csv: Path
    image_dir: Path


class ClassificationMetrics(TypedDict):
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    f1_AD: float
    f1_HP: float
    f1_ASS: float
    recall_AD: float
    recall_HP: float
    recall_ASS: float


@dataclass(frozen=True)
class ClassificationModelResult:
    model: str
    params: dict[str, object]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    f1_AD: float
    f1_HP: float
    f1_ASS: float
    recall_AD: float
    recall_HP: float
    recall_ASS: float
    valid_predictions: np.ndarray

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row.pop("valid_predictions", None)
        return row


BenchmarkPaths = type(COMMON_PATHS)
PROJECT_ROOT = COMMON_PROJECT_ROOT
RAW_ROOT = COMMON_RAW_ROOT
TRAIN_ROOT = COMMON_TRAIN_ROOT
VALID_ROOT = COMMON_VALID_ROOT
TEST_ROOT = COMMON_TEST_ROOT
TRAIN_CSV = COMMON_TRAIN_CSV
VALID_CSV = COMMON_VALID_CSV
TEST_CSV = COMMON_TEST_CSV
TRAIN_IMAGE_DIR = COMMON_TRAIN_IMAGE_DIR
VALID_IMAGE_DIR = COMMON_VALID_IMAGE_DIR
TEST_IMAGE_DIR = COMMON_TEST_IMAGE_DIR
TRAIN_MASK_DIR = COMMON_TRAIN_MASK_DIR
VALID_MASK_DIR = COMMON_VALID_MASK_DIR
PATHS = COMMON_PATHS
SPLIT_SPECS = COMMON_SPLIT_SPECS
EXPORT_DIR = PROJECT_ROOT / "data" / "experiments" / "polyp_classification"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


find_project_root = common_find_project_root
find_by_stem = common_find_by_stem
load_image_bgr = common_load_image_bgr
load_mask = common_load_mask
overlay_mask = common_overlay_mask
load_split_records = common_load_split_records


def resize_image(image_bgr: np.ndarray, *, max_side: int = 128) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(float(max_side) / float(max(height, width)), 1.0)
    if scale == 1.0:
        return image_bgr
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def image_feature_vector(image_bgr: np.ndarray, *, max_side: int = 128) -> np.ndarray:
    resized = resize_image(image_bgr, max_side=max_side)
    resized_float = resized.astype(np.float32) / 255.0
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1:] /= 255.0

    channels = [resized_float[:, :, idx] for idx in range(3)]
    summary_features = []
    for channel in channels + [gray, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]:
        summary_features.extend([float(channel.mean()), float(channel.std()), float(channel.min()), float(channel.max())])

    hist_features: list[float] = []
    for channel in channels:
        histogram = cv2.calcHist([np.clip(channel * 255.0, 0, 255).astype(np.uint8)], [0], None, [16], [0, 256]).reshape(-1)
        histogram = histogram / max(float(histogram.sum()), 1.0)
        hist_features.extend(histogram.astype(float).tolist())

    if skimage_hog is not None:
        gray_resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        hog_vector = skimage_hog(
            gray_resized,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        hog_features = hog_vector.astype(float).tolist()[:64]
    else:
        hog_features = []

    flat_gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1).astype(float).tolist()

    return np.array(summary_features + hist_features + hog_features + flat_gray, dtype=np.float32)


def build_feature_table(split_df: pd.DataFrame) -> pd.DataFrame:
    feature_rows: list[dict[str, object]] = []
    for record in split_df.to_dict(orient="records"):
        image = load_image_bgr(Path(str(record["image_path"])))
        features = image_feature_vector(image)
        row: dict[str, object] = {
            "split": str(record["split"]),
            "image_id": str(record["image_id"]),
            "cls": str(record["cls"]),
            "Histologia": record["Histologia"],
        }
        for idx, value in enumerate(features):
            row[f"f_{idx:04d}"] = float(value)
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def build_estimator(model_name: str, params: Mapping[str, object]) -> Any:
    if model_name == "Dummy majority":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "Logistic Regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                C=float(cast(Any, params.get("C", 1.0))),
            ),
        )
    if model_name == "SVM RBF":
        gamma_value = cast(Any, params.get("gamma", "scale"))
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                C=float(cast(Any, params.get("C", 1.0))),
                gamma=gamma_value,
                random_state=RANDOM_STATE,
            ),
        )
    if model_name == "Random Forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=int(cast(Any, params.get("n_estimators", 400))),
                max_depth=cast(Optional[int], params.get("max_depth")),
                min_samples_leaf=int(cast(Any, params.get("min_samples_leaf", 1))),
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        )
    if model_name == "k-NN":
        weights_value = cast(Any, params.get("weights", "distance"))
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(cast(Any, params.get("n_neighbors", 7))),
                weights=weights_value,
            ),
        )
    if model_name == "Gradient Boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(
                n_estimators=int(cast(Any, params.get("n_estimators", 200))),
                learning_rate=float(cast(Any, params.get("learning_rate", 0.05))),
                subsample=float(cast(Any, params.get("subsample", 1.0))),
                random_state=RANDOM_STATE,
            ),
        )
    raise KeyError(model_name)


def summarize_predictions(y_true: pd.Series, y_pred: np.ndarray) -> ClassificationMetrics:
    report_dict = cast(dict[str, Mapping[str, float]], classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        **{f"f1_{cls}": float(report_dict[cls]["f1-score"]) for cls in CLASS_ORDER},
        **{f"recall_{cls}": float(report_dict[cls]["recall"]) for cls in CLASS_ORDER},
    }
    return cast(ClassificationMetrics, metrics)


candidate_params: dict[str, list[dict[str, object]]] = {
    "Dummy majority": [{}],
    "Logistic Regression": [{"C": 0.25}, {"C": 1.0}, {"C": 3.0}, {"C": 8.0}],
    "SVM RBF": [
        {"C": 0.75, "gamma": "scale"},
        {"C": 1.5, "gamma": "scale"},
        {"C": 3.0, "gamma": "scale"},
        {"C": 6.0, "gamma": "auto"},
    ],
    "Random Forest": [
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 700, "max_depth": 22, "min_samples_leaf": 1},
        {"n_estimators": 1000, "max_depth": 26, "min_samples_leaf": 2},
    ],
    "k-NN": [
        {"n_neighbors": 3, "weights": "distance"},
        {"n_neighbors": 5, "weights": "distance"},
        {"n_neighbors": 7, "weights": "distance"},
        {"n_neighbors": 11, "weights": "distance"},
    ],
    "Gradient Boosting": [
        {"n_estimators": 150, "learning_rate": 0.05, "subsample": 0.85},
        {"n_estimators": 250, "learning_rate": 0.04, "subsample": 0.85},
        {"n_estimators": 300, "learning_rate": 0.03, "subsample": 0.9},
    ],
}


def evaluate_classification_models(train_features: pd.DataFrame, valid_features: pd.DataFrame, feature_cols: Sequence[str]) -> list[ClassificationModelResult]:
    X_train = train_features[list(feature_cols)].to_numpy()
    y_train = train_features["cls"].to_numpy()
    X_valid = valid_features[list(feature_cols)].to_numpy()
    y_valid = valid_features["cls"].to_numpy()

    results: list[ClassificationModelResult] = []
    for model_name, param_grid in candidate_params.items():
        for params in param_grid:
            estimator = build_estimator(model_name, params)
            estimator.fit(X_train, y_train)
            valid_predictions = estimator.predict(X_valid)
            metrics = summarize_predictions(pd.Series(y_valid), valid_predictions)
            results.append(
                ClassificationModelResult(
                    model=model_name,
                    params=dict(params),
                    accuracy=metrics["accuracy"],
                    balanced_accuracy=metrics["balanced_accuracy"],
                    macro_f1=metrics["macro_f1"],
                    weighted_f1=metrics["weighted_f1"],
                    f1_AD=metrics["f1_AD"],
                    f1_HP=metrics["f1_HP"],
                    f1_ASS=metrics["f1_ASS"],
                    recall_AD=metrics["recall_AD"],
                    recall_HP=metrics["recall_HP"],
                    recall_ASS=metrics["recall_ASS"],
                    valid_predictions=valid_predictions,
                )
            )

    return results


def best_classification_results(results: Sequence[ClassificationModelResult]) -> dict[str, ClassificationModelResult]:
    best: dict[str, ClassificationModelResult] = {}
    for result in results:
        current_best = best.get(result.model)
        if current_best is None or result.macro_f1 > current_best.macro_f1:
            best[result.model] = result
    return best


def classification_results_frame(results: Sequence[ClassificationModelResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_row() for result in results])


def render_classification_confusion_matrices(
    y_true: Sequence[str],
    results: Sequence[ClassificationModelResult],
    *,
    title_prefix: str = "Validación",
) -> None:
    if not results:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(results) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5 * n_cols, 5.5 * n_rows))
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, result in zip(axes_array, results, strict=False):
        cm = confusion_matrix(y_true, result.valid_predictions, labels=CLASS_ORDER)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axis)
        axis.set_title(f"{result.model} | {title_prefix}\nparams={result.params}")
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Real")

    for axis in axes_array[len(results):]:
        axis.remove()

    plt.tight_layout()


def render_random_classification_examples(
    model_name: str,
    estimator: Any,
    split_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    numeric_feature_cols: list[str],
    *,
    n_samples: int = 2,
) -> None:
    if split_df.empty:
        return

    sample_count = min(n_samples, len(split_df))
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(len(split_df), size=sample_count, replace=False)
    fig, axes = plt.subplots(sample_count, 1, figsize=(6, 4.5 * sample_count))
    axes = np.atleast_1d(axes)

    for row_idx, sample_idx in enumerate(sample_indices):
        split_row = split_df.iloc[int(sample_idx)]
        feature_row = feature_df.iloc[int(sample_idx)]
        image = load_image_bgr(Path(str(split_row["image_path"])))
        predicted_class = estimator.predict(np.asarray(feature_row[numeric_feature_cols].to_numpy(dtype=np.float32)).reshape(1, -1))[0]
        actual_class = str(split_row["cls"])

        axes[row_idx].imshow(image[:, :, ::-1])
        axes[row_idx].set_title(f"Real={actual_class} | Pred={predicted_class} | ID={split_row['image_id']}")
        axes[row_idx].axis("off")

    fig.suptitle(f"Modelo: {model_name}", y=1.01)
    plt.tight_layout()
