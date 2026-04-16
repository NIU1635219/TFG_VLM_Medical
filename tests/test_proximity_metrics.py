from __future__ import annotations

import pytest

from src.utils.tests_ui.metrics import (
    calculate_center_distance_score,
    calculate_proximity_score,
    calculate_size_relative_score,
)


def test_center_distance_score_identical_boxes_is_one() -> None:
    box = [100, 100, 300, 400]
    assert calculate_center_distance_score(box, box) == pytest.approx(1.0)


def test_center_distance_score_decreases_for_far_boxes() -> None:
    box_a = [0, 0, 100, 100]
    box_b = [900, 900, 1000, 1000]

    score = calculate_center_distance_score(box_a, box_b)
    assert 0.0 <= score < 0.2


def test_size_relative_score_identical_area_is_one() -> None:
    box_a = [100, 100, 300, 300]
    box_b = [200, 200, 400, 400]

    assert calculate_size_relative_score(box_a, box_b) == pytest.approx(1.0)


def test_size_relative_score_half_area_gap() -> None:
    box_a = [100, 100, 300, 300]  # area 40_000
    box_b = [100, 100, 500, 500]  # area 160_000

    score = calculate_size_relative_score(box_a, box_b)
    assert score == pytest.approx(0.25)


def test_proximity_score_returns_components_and_combined() -> None:
    box_a = [100, 100, 350, 350]
    box_b = [120, 120, 370, 370]

    payload = calculate_proximity_score(box_a, box_b)

    assert set(payload.keys()) == {
        "proximity_score",
        "proximity_center_score",
        "proximity_size_score",
    }
    assert 0.0 <= payload["proximity_score"] <= 1.0
    assert 0.0 <= payload["proximity_center_score"] <= 1.0
    assert 0.0 <= payload["proximity_size_score"] <= 1.0


def test_proximity_score_invalid_box_raises() -> None:
    box_a = [500, 100, 300, 200]  # xmin > xmax
    box_b = [100, 100, 200, 200]

    with pytest.raises(ValueError):
        calculate_proximity_score(box_a, box_b)


def test_proximity_score_is_near_zero_when_boxes_do_not_touch() -> None:
    box_a = [0, 0, 100, 100]
    box_b = [300, 300, 400, 400]

    payload = calculate_proximity_score(box_a, box_b)

    assert payload["proximity_score"] < 0.05


def test_proximity_score_favors_containment_over_partial_overlap() -> None:
    gt_box = [100, 100, 300, 300]
    pred_contained = [140, 140, 260, 260]
    pred_partial = [220, 220, 360, 360]

    score_contained = calculate_proximity_score(gt_box, pred_contained)["proximity_score"]
    score_partial = calculate_proximity_score(gt_box, pred_partial)["proximity_score"]

    assert score_contained > score_partial
