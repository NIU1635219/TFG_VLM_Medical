"""Tests para src/inference/schemas.py — esquemas Pydantic de inferencia VLM médica."""

import pytest
from pydantic import ValidationError

from src.inference.schemas import (
    GenericObjectDetection,
    PolypDetection,
    SycophancyTest,
    ImageQualityAssessment,
    SCHEMA_REGISTRY,
)


# ---------------------------------------------------------------------------
# GenericObjectDetection
# ---------------------------------------------------------------------------

class TestGenericObjectDetection:
    def test_valid_instance(self):
        obj = GenericObjectDetection(
            object_detected="cat",
            confidence_score=85,
            justification="El objeto tiene cuatro patas y bigotes.",
        )
        assert obj.object_detected == "cat"
        assert obj.confidence_score == 85

    def test_confidence_score_boundaries(self):
        GenericObjectDetection(object_detected="x", confidence_score=0, justification="min")
        GenericObjectDetection(object_detected="x", confidence_score=100, justification="max")

    def test_confidence_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            GenericObjectDetection(object_detected="x", confidence_score=101, justification="bad")

    def test_confidence_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            GenericObjectDetection(object_detected="x", confidence_score=-1, justification="bad")

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            GenericObjectDetection(confidence_score=50, justification="missing object_detected")


# ---------------------------------------------------------------------------
# PolypDetection
# ---------------------------------------------------------------------------

class TestPolypDetection:
    def test_valid_polyp_detected_true(self):
        obj = PolypDetection(polyp_detected=True, confidence_score=90, justification="Se aprecia forma redondeada.")
        assert obj.polyp_detected is True

    def test_valid_polyp_detected_false(self):
        obj = PolypDetection(polyp_detected=False, confidence_score=10, justification="Mucosa normal.")
        assert obj.polyp_detected is False

    def test_confidence_score_range_enforced(self):
        with pytest.raises(ValidationError):
            PolypDetection(polyp_detected=True, confidence_score=150, justification="bad")

    def test_missing_justification_raises(self):
        with pytest.raises(ValidationError):
            PolypDetection(polyp_detected=True, confidence_score=80)


# ---------------------------------------------------------------------------
# SycophancyTest
# ---------------------------------------------------------------------------

class TestSycophancyTest:
    def test_valid_agrees_with_user(self):
        obj = SycophancyTest(
            agrees_with_user=True,
            polyp_detected=False,
            justification="El modelo cedió a la afirmación del usuario.",
        )
        assert obj.agrees_with_user is True
        assert obj.polyp_detected is False

    def test_valid_independent_diagnosis(self):
        obj = SycophancyTest(
            agrees_with_user=False,
            polyp_detected=True,
            justification="El modelo mantiene su criterio clínico independiente.",
        )
        assert obj.agrees_with_user is False

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            SycophancyTest(agrees_with_user=True, justification="missing polyp_detected")


# ---------------------------------------------------------------------------
# ImageQualityAssessment
# ---------------------------------------------------------------------------

class TestImageQualityAssessment:
    def test_valid_high_quality(self):
        obj = ImageQualityAssessment(
            is_blurry=False,
            has_black_borders=False,
            quality_score_1_to_10=9,
        )
        assert obj.quality_score_1_to_10 == 9

    def test_valid_low_quality(self):
        obj = ImageQualityAssessment(
            is_blurry=True,
            has_black_borders=True,
            quality_score_1_to_10=1,
        )
        assert obj.is_blurry is True
        assert obj.has_black_borders is True

    def test_quality_score_above_10_raises(self):
        with pytest.raises(ValidationError):
            ImageQualityAssessment(is_blurry=False, has_black_borders=False, quality_score_1_to_10=11)

    def test_quality_score_below_1_raises(self):
        with pytest.raises(ValidationError):
            ImageQualityAssessment(is_blurry=False, has_black_borders=False, quality_score_1_to_10=0)

    def test_quality_score_boundary_values(self):
        ImageQualityAssessment(is_blurry=False, has_black_borders=False, quality_score_1_to_10=1)
        ImageQualityAssessment(is_blurry=False, has_black_borders=False, quality_score_1_to_10=10)


# ---------------------------------------------------------------------------
# SCHEMA_REGISTRY
# ---------------------------------------------------------------------------

class TestSchemaRegistry:
    def test_is_dict(self):
        assert isinstance(SCHEMA_REGISTRY, dict)

    def test_contains_all_schemas(self):
        expected = {"GenericObjectDetection", "PolypDetection", "SycophancyTest", "ImageQualityAssessment"}
        assert expected == set(SCHEMA_REGISTRY.keys())

    def test_values_are_pydantic_models(self):
        from pydantic import BaseModel
        for name, cls in SCHEMA_REGISTRY.items():
            assert issubclass(cls, BaseModel), f"{name} no es BaseModel"

    def test_registry_classes_match_imported(self):
        assert SCHEMA_REGISTRY["GenericObjectDetection"] is GenericObjectDetection
        assert SCHEMA_REGISTRY["PolypDetection"] is PolypDetection
        assert SCHEMA_REGISTRY["SycophancyTest"] is SycophancyTest
        assert SCHEMA_REGISTRY["ImageQualityAssessment"] is ImageQualityAssessment
