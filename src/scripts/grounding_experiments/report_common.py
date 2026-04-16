"""Common helper functions shared across grounding experiment modules."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_polyp_class(value: Any) -> str:
    """Normalize polyp class labels to stable uppercase text."""
    return str(value or "").strip().upper()


def extract_predicted_class(payload: Mapping[str, Any], fallback: Any = "") -> str:
    """Return predicted class using final_diagnosis_class as canonical key."""
    value = payload.get("final_diagnosis_class")
    if isinstance(value, str):
        return value.strip()
    if isinstance(fallback, str):
        return fallback.strip()
    return ""


def build_bbox_xyxy(*, xmin: Any, ymin: Any, xmax: Any, ymax: Any) -> list[Any]:
    """Build bbox list in canonical [xmin, ymin, xmax, ymax] order."""
    return [xmin, ymin, xmax, ymax]


def build_bbox_xyxy_from_mapping(mapping: Mapping[str, Any]) -> list[Any]:
    """Extract bbox from mapping keys using canonical [xmin, ymin, xmax, ymax]."""
    return build_bbox_xyxy(
        xmin=mapping.get("xmin"),
        ymin=mapping.get("ymin"),
        xmax=mapping.get("xmax"),
        ymax=mapping.get("ymax"),
    )
