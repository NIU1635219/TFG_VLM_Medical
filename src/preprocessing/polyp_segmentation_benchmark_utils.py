from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing.polyp_benchmark_utils import (
    PROJECT_ROOT,
    RAW_ROOT,
    TEST_CSV,
    TEST_IMAGE_DIR,
    TEST_ROOT,
    TRAIN_CSV,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    TRAIN_ROOT,
    VALID_CSV,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    VALID_ROOT,
    find_by_stem,
    find_project_root,
    load_image_bgr,
    load_mask,
    load_split_records,
    overlay_mask,
)
from src.preprocessing.mask_generation import generate_mask_from_image

SEGMENTATION_MAX_SIDE = 256
SEGMENTATION_LABELS = (0, 1)
RANDOM_STATE = 42
EXPORT_DIR = PROJECT_ROOT / "data" / "experiments" / "polyp_segmentation"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SegmentationModelResult:
    model: str
    params: dict[str, object]
    dice: float
    iou: float
    precision: float
    recall: float
    accuracy: float
    estimator: Any

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row.pop("estimator", None)
        return row


candidate_params: dict[str, list[dict[str, object]]] = {
    "Dummy majority": [{}],
    "Logistic Regression": [{"C": 0.25}, {"C": 1.0}, {"C": 3.0}],
    "Linear SVM": [{"C": 0.1}, {"C": 1.0}, {"C": 3.0}],
    "Random Forest": [
        {"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 2},
        {"n_estimators": 250, "max_depth": 20, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 24, "min_samples_leaf": 1},
    ],
    "Gradient Boosting": [
        {"n_estimators": 150, "learning_rate": 0.05, "subsample": 0.85},
        {"n_estimators": 250, "learning_rate": 0.04, "subsample": 0.85},
    ],
}


def resize_to_max_side(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    max_side: int = SEGMENTATION_MAX_SIDE,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    height, width = image_bgr.shape[:2]
    scale = min(float(max_side) / float(max(height, width)), 1.0)
    if scale == 1.0:
        return image_bgr, mask, 1.0

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_image = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = None
    if mask is not None:
        resized_mask = cv2.resize(mask.astype(np.uint8), (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 0).astype(np.uint8)
    return resized_image, resized_mask, scale


def pixel_feature_grid(image_bgr: np.ndarray) -> np.ndarray:
    image_float = image_bgr.astype(np.float32) / 255.0
    hsv_float = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_float[:, :, 0] /= 179.0
    hsv_float[:, :, 1:] /= 255.0

    height, width = image_bgr.shape[:2]
    ys, xs = np.indices((height, width))
    x_norm = xs.astype(np.float32) / max(width - 1, 1)
    y_norm = ys.astype(np.float32) / max(height - 1, 1)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    features = np.dstack(
        [
            image_float[:, :, 0],
            image_float[:, :, 1],
            image_float[:, :, 2],
            hsv_float[:, :, 0],
            hsv_float[:, :, 1],
            hsv_float[:, :, 2],
            gray,
            x_norm,
            y_norm,
        ]
    )
    return features.reshape(-1, features.shape[-1])


def sample_pixels_from_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    fg_pixels: int = 128,
    bg_pixels: int = 128,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    resized_image, resized_mask, _ = resize_to_max_side(image_bgr, mask, max_side=SEGMENTATION_MAX_SIDE)
    assert resized_mask is not None
    features = pixel_feature_grid(resized_image)
    labels = resized_mask.reshape(-1).astype(np.uint8)

    rng = np.random.default_rng(random_state)
    fg_indices = np.flatnonzero(labels == 1)
    bg_indices = np.flatnonzero(labels == 0)

    fg_take = min(fg_pixels, len(fg_indices))
    bg_take = min(bg_pixels, len(bg_indices))
    if fg_take == 0 or bg_take == 0:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    sampled_fg = rng.choice(fg_indices, size=fg_take, replace=False)
    sampled_bg = rng.choice(bg_indices, size=bg_take, replace=False)
    sampled_indices = np.concatenate([sampled_fg, sampled_bg])
    rng.shuffle(sampled_indices)

    return features[sampled_indices], labels[sampled_indices]


def build_segmentation_dataset(
    split_df: pd.DataFrame,
    *,
    fg_pixels_per_image: int = 128,
    bg_pixels_per_image: int = 128,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    for offset, record in enumerate(split_df.to_dict(orient="records")):
        image = load_image_bgr(Path(str(record["image_path"])))
        mask = load_mask(Path(str(record["mask_path"])))
        x_part, y_part = sample_pixels_from_image(
            image,
            mask,
            fg_pixels=fg_pixels_per_image,
            bg_pixels=bg_pixels_per_image,
            random_state=random_state + offset,
        )
        if x_part.size == 0:
            continue
        x_chunks.append(x_part)
        y_chunks.append(y_part)

    if not x_chunks:
        return np.empty((0, 9), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    return np.vstack(x_chunks), np.concatenate(y_chunks)


def build_estimator(model_name: str, params: Mapping[str, object]) -> Any:
    if model_name == "Dummy majority":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "Logistic Regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                C=float(cast(Any, params.get("C", 1.0))),
            ),
        )
    if model_name == "Linear SVM":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                C=float(cast(Any, params.get("C", 1.0))),
                random_state=RANDOM_STATE,
            ),
        )
    if model_name == "Random Forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=int(cast(Any, params.get("n_estimators", 250))),
                max_depth=cast(Optional[int], params.get("max_depth")),
                min_samples_leaf=int(cast(Any, params.get("min_samples_leaf", 1))),
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    if model_name == "Gradient Boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(
                n_estimators=int(cast(Any, params.get("n_estimators", 200))),
                learning_rate=float(cast(Any, params.get("learning_rate", 0.05))),
                subsample=float(cast(Any, params.get("subsample", 0.85))),
                random_state=RANDOM_STATE,
            ),
        )
    raise KeyError(model_name)


def evaluate_segmentation_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    fg_pixels_per_image: int = 160,
    bg_pixels_per_image: int = 160,
) -> list[SegmentationModelResult]:
    X_train, y_train = build_segmentation_dataset(
        train_df,
        fg_pixels_per_image=fg_pixels_per_image,
        bg_pixels_per_image=bg_pixels_per_image,
    )

    results: list[SegmentationModelResult] = []
    for model_name, param_grid in candidate_params.items():
        for params in param_grid:
            estimator = build_estimator(model_name, params)
            estimator.fit(X_train, y_train)
            valid_model_df = evaluate_segmentation_model(estimator, valid_df)
            results.append(
                SegmentationModelResult(
                    model=model_name,
                    params=dict(params),
                    dice=float(valid_model_df["dice"].mean()),
                    iou=float(valid_model_df["iou"].mean()),
                    precision=float(valid_model_df["precision"].mean()),
                    recall=float(valid_model_df["recall"].mean()),
                    accuracy=float(valid_model_df["accuracy"].mean()),
                    estimator=estimator,
                )
            )

    return results


def best_segmentation_result(results: Sequence[SegmentationModelResult]) -> SegmentationModelResult:
    if not results:
        raise ValueError("No hay resultados de segmentación")
    return max(results, key=lambda result: result.dice)


def best_segmentation_results(results: Sequence[SegmentationModelResult]) -> dict[str, SegmentationModelResult]:
    best: dict[str, SegmentationModelResult] = {}
    for result in results:
        current_best = best.get(result.model)
        if current_best is None or result.dice > current_best.dice:
            best[result.model] = result
    return best


def segmentation_results_frame(results: Sequence[SegmentationModelResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_row() for result in results])


def summarize_mask_prediction(y_true_mask: np.ndarray, y_pred_mask: np.ndarray) -> dict[str, float]:
    y_true_flat = y_true_mask.reshape(-1).astype(np.uint8)
    y_pred_flat = y_pred_mask.reshape(-1).astype(np.uint8)
    return {
        "accuracy": float(accuracy_score(y_true_flat, y_pred_flat)),
        "dice": float(f1_score(y_true_flat, y_pred_flat, zero_division=0)),
        "iou": float(jaccard_score(y_true_flat, y_pred_flat, zero_division=0)),
        "precision": float(precision_score(y_true_flat, y_pred_flat, zero_division=0)),
        "recall": float(recall_score(y_true_flat, y_pred_flat, zero_division=0)),
    }


def predict_mask_from_estimator(
    estimator: Any,
    image_bgr: np.ndarray,
    *,
    max_side: int = SEGMENTATION_MAX_SIDE,
    threshold: float = 0.5,
) -> np.ndarray:
    resized_image, _, _ = resize_to_max_side(image_bgr, None, max_side=max_side)
    feature_grid = pixel_feature_grid(resized_image)

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(feature_grid)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            foreground_scores = probabilities[:, 1]
        else:
            foreground_scores = probabilities.reshape(-1)
        predicted_flat = (foreground_scores >= threshold).astype(np.uint8)
    elif hasattr(estimator, "decision_function"):
        decision_scores = estimator.decision_function(feature_grid)
        if np.ndim(decision_scores) > 1:
            decision_scores = decision_scores[:, 0]
        predicted_flat = (np.asarray(decision_scores) >= 0.0).astype(np.uint8)
    else:
        predicted_flat = estimator.predict(feature_grid).astype(np.uint8)

    return predicted_flat.reshape(resized_image.shape[:2])


def evaluate_segmentation_model(
    estimator: Any,
    split_df: pd.DataFrame,
    *,
    max_side: int = SEGMENTATION_MAX_SIDE,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in split_df.to_dict(orient="records"):
        image = load_image_bgr(Path(str(record["image_path"])))
        mask = load_mask(Path(str(record["mask_path"])))
        resized_image, resized_mask, _ = resize_to_max_side(image, mask, max_side=max_side)
        assert resized_mask is not None
        predicted_mask = predict_mask_from_estimator(estimator, resized_image, max_side=max_side)
        metrics = summarize_mask_prediction(resized_mask, predicted_mask)
        rows.append({"image_id": record["image_id"], "cls": record["cls"], **metrics})
    return pd.DataFrame(rows)


def render_segmentation_examples(
    estimator: Any,
    split_df: pd.DataFrame,
    *,
    n_samples: int = 2,
    show_gt: bool = True,
    max_side: int = SEGMENTATION_MAX_SIDE,
    model_name: Optional[str] = None,
) -> None:
    if split_df.empty:
        return

    sample_count = min(n_samples, len(split_df))
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(len(split_df), size=sample_count, replace=False)
    n_cols = 3 if show_gt else 2
    fig, axes = plt.subplots(sample_count, n_cols, figsize=(5 * n_cols, 5 * sample_count))
    axes = np.atleast_2d(axes)
    model_label = model_name or estimator.__class__.__name__

    fig.suptitle(f"Modelo: {model_label}", y=1.01)

    for row_idx, sample_idx in enumerate(sample_indices):
        record = split_df.iloc[int(sample_idx)]
        image = load_image_bgr(Path(str(record["image_path"])))
        resized_image, resized_mask, _ = resize_to_max_side(image, load_mask(Path(str(record["mask_path"]))) if show_gt else None, max_side=max_side)
        predicted_mask = predict_mask_from_estimator(estimator, image, max_side=max_side)

        axes[row_idx, 0].imshow(resized_image[:, :, ::-1])
        axes[row_idx, 0].set_title(f"Real={record['cls']} | ID={record['image_id']}")
        axes[row_idx, 0].axis("off")

        if show_gt and resized_mask is not None:
            gt_overlay = overlay_mask(resized_image, resized_mask)
            axes[row_idx, 1].imshow(gt_overlay[:, :, ::-1])
            axes[row_idx, 1].set_title("Ground truth")
            axes[row_idx, 1].axis("off")
            pred_overlay = overlay_mask(resized_image, predicted_mask)
            axes[row_idx, 2].imshow(pred_overlay[:, :, ::-1])
            axes[row_idx, 2].set_title("Predicción del modelo")
            axes[row_idx, 2].axis("off")
        else:
            pred_overlay = overlay_mask(resized_image, predicted_mask)
            axes[row_idx, 1].imshow(pred_overlay[:, :, ::-1])
            axes[row_idx, 1].set_title("Predicción del modelo")
            axes[row_idx, 1].axis("off")

    plt.tight_layout()


def render_segmentation_examples_by_model(
    results: Sequence[SegmentationModelResult],
    split_df: pd.DataFrame,
    *,
    n_samples: int = 3,
    show_gt: bool = True,
) -> None:
    for result in results:
        print(f"Modelo: {result.model} | params={result.params} | dice={result.dice:.4f} | iou={result.iou:.4f}")
        render_segmentation_examples(
            result.estimator,
            split_df,
            n_samples=n_samples,
            show_gt=show_gt,
            model_name=result.model,
        )
