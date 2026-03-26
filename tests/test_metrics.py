from __future__ import annotations

import pytest

from src.utils.tests_ui.metrics import calculate_iou


def test_iou_perfect_match() -> None:
    box_a = [100, 200, 800, 900]
    box_b = [100, 200, 800, 900]

    assert calculate_iou(box_a, box_b) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    box_a = [0, 0, 100, 100]
    box_b = [200, 200, 300, 300]

    assert calculate_iou(box_a, box_b) == pytest.approx(0.0)


def test_iou_partial_overlap() -> None:
    box_a = [100, 100, 500, 500]
    box_b = [300, 300, 700, 700]

    # Interseccion = 200x200 = 40_000; Union = 160_000 + 160_000 - 40_000 = 280_000
    expected_iou = 40_000 / 280_000

    assert calculate_iou(box_a, box_b) == pytest.approx(expected_iou)


def test_iou_concentric_boxes() -> None:
    box_a = [100, 100, 900, 900]
    box_b = [300, 300, 500, 500]

    # Caja pequena dentro de caja grande: IoU = area_small / area_big = 40_000 / 640_000
    expected_iou = 40_000 / 640_000

    assert calculate_iou(box_a, box_b) == pytest.approx(expected_iou)


def test_iou_invalid_box_raises_error() -> None:
    box_a = [900, 100, 800, 500]  # ymin > ymax
    box_b = [100, 100, 800, 500]

    with pytest.raises(ValueError):
        calculate_iou(box_a, box_b)
