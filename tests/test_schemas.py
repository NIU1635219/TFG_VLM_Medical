"""Tests para src/inference/schemas.py — esquemas Pydantic de inferencia VLM médica."""

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.inference.schemas import (
    AdvancedPolypClassification,
    AdvancedPolypClassificationWithReasoning,
    ClassEvidence,
    GenericObjectDetection,
    GenericObjectDetectionWithReasoning,
    PolypClassification,
    PolypClassificationWithReasoning,
    PolypDetection,
    PolypDetectionWithReasoning,
    SycophancyTest,
    SycophancyTestWithReasoning,
    ImageQualityAssessment,
    ImageQualityAssessmentWithReasoning,
    REASONING_SCHEMA_REGISTRY,
    SCHEMA_REGISTRY,
    _create_reasoning_schema,
    get_schema_variant,
    schema_uses_reasoning,
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
        obj = PolypDetection(
            polyp_detected=True,
            confidence_score=90,
            justification="La morfología sobreelevada y la textura focal son compatibles con pólipo.",
        )
        assert obj.polyp_detected is True

    def test_valid_polyp_detected_false(self):
        obj = PolypDetection(
            polyp_detected=False,
            confidence_score=10,
            justification="No se identifican hallazgos sugestivos de pólipo.",
        )
        assert obj.polyp_detected is False

    def test_confidence_score_range_enforced(self):
        with pytest.raises(ValidationError):
            PolypDetection(
                polyp_detected=True,
                confidence_score=150,
                justification="bad",
            )

    def test_missing_justification_raises(self):
        with pytest.raises(ValidationError):
            PolypDetection(
                polyp_detected=True,
                confidence_score=80,
            )


class TestPolypClassification:
    def test_valid_histology_class(self):
        obj = PolypClassification(
            predicted_class="AD",
            confidence_score=87,
            justification="Patrón compatible con adenoma.",
        )
        assert obj.predicted_class == "AD"

    def test_rejects_invalid_histology_class(self):
        with pytest.raises(ValidationError):
            PolypClassification(
                predicted_class="INVALID",
                confidence_score=70,
                justification="bad",
            )

    def test_missing_justification_raises(self):
        with pytest.raises(ValidationError):
            PolypClassification(
                predicted_class="HP",
                confidence_score=64,
            )


class TestAdvancedPolypClassification:
    def test_class_evidence_requires_both_fields(self):
        with pytest.raises(ValidationError):
            ClassEvidence(evidence_for="Compatible con arquitectura serrada")

    def test_valid_advanced_polyp_classification(self):
        obj = AdvancedPolypClassification(
            analysis_ad=ClassEvidence(
                evidence_for="Patrón glandular compatible con adenoma.",
                evidence_against="No se observan criterios fuertes de AD en toda la lesión.",
            ),
            analysis_hp=ClassEvidence(
                evidence_for="Coloración homogénea sugerente de HP.",
                evidence_against="La arquitectura superficial no encaja con HP clásico.",
            ),
            analysis_ass=ClassEvidence(
                evidence_for="Bordes mal definidos y patrón serrado compatible con ASS.",
                evidence_against="No hay moco evidente en toda la superficie.",
            ),
            clinical_consensus="Predomina evidencia morfológica de ASS sobre AD y HP.",
            final_diagnosis="ASS",
        )

        assert obj.final_diagnosis == "ASS"
        assert obj.analysis_ass.evidence_for

    def test_rejects_invalid_final_diagnosis(self):
        with pytest.raises(ValidationError):
            AdvancedPolypClassification(
                analysis_ad=ClassEvidence(evidence_for="a", evidence_against="b"),
                analysis_hp=ClassEvidence(evidence_for="a", evidence_against="b"),
                analysis_ass=ClassEvidence(evidence_for="a", evidence_against="b"),
                clinical_consensus="c",
                final_diagnosis="UNKNOWN",
            )

    def test_rejects_indeterminado_final_diagnosis(self):
        with pytest.raises(ValidationError):
            AdvancedPolypClassification(
                analysis_ad=ClassEvidence(evidence_for="a", evidence_against="b"),
                analysis_hp=ClassEvidence(evidence_for="a", evidence_against="b"),
                analysis_ass=ClassEvidence(evidence_for="a", evidence_against="b"),
                clinical_consensus="Imagen atipica sin confianza suficiente para clase cerrada.",
                final_diagnosis="INDETERMINADO",
            )

    def test_exposes_default_system_prompt(self):
        prompt = AdvancedPolypClassification.DEFAULT_SYSTEM_PROMPT
        assert isinstance(prompt, str)
        assert "AD" in prompt and "HP" in prompt and "ASS" in prompt
        assert "COLONOSCOPIAS" in prompt
        assert "EXISTE una lesion" in prompt


class TestReasoningSchemaVariants:
    def test_reasoning_registry_contains_all_base_schemas(self):
        expected = {
            "GenericObjectDetection",
            "PolypDetection",
            "PolypClassification",
            "AdvancedPolypClassification",
            "SycophancyTest",
            "ImageQualityAssessment",
        }
        assert expected == set(REASONING_SCHEMA_REGISTRY.keys())

    def test_get_schema_variant_returns_base_schema(self):
        schema_name, schema_cls = get_schema_variant("PolypDetection", include_reasoning=False)
        assert schema_name == "PolypDetection"
        assert schema_cls is PolypDetection

    def test_get_schema_variant_returns_reasoning_schema(self):
        schema_name, schema_cls = get_schema_variant("PolypDetection", include_reasoning=True)
        assert schema_name == "PolypDetectionWithReasoning"
        assert schema_cls is PolypDetectionWithReasoning

    def test_schema_uses_reasoning_detects_variants(self):
        assert schema_uses_reasoning(PolypDetection) is False
        assert schema_uses_reasoning(PolypDetectionWithReasoning) is True

    def test_reasoning_variant_declares_reasoning_field(self):
        assert "reasoning" in GenericObjectDetectionWithReasoning.model_fields

    def test_reasoning_variant_json_schema_declares_reasoning_first(self):
        property_names = list(GenericObjectDetectionWithReasoning.model_json_schema()["properties"].keys())
        assert property_names[0] == "reasoning"

    def test_advanced_reasoning_variant_places_reasoning_before_clinical_consensus(self):
        property_names = list(
            AdvancedPolypClassificationWithReasoning.model_json_schema()["properties"].keys()
        )
        assert property_names.index("analysis_ass") < property_names.index("reasoning")
        assert property_names.index("reasoning") < property_names.index("clinical_consensus")

    def test_advanced_reasoning_variant_accepts_general_reasoning_text(self):
        obj = AdvancedPolypClassificationWithReasoning(
            analysis_ad=ClassEvidence(evidence_for="a", evidence_against="b"),
            analysis_hp=ClassEvidence(evidence_for="a", evidence_against="b"),
            analysis_ass=ClassEvidence(evidence_for="a", evidence_against="b"),
            reasoning="Lesion sutil con evidencia mixta; tras comparar AD/HP/ASS, predomina ASS.",
            clinical_consensus="La evidencia global favorece ASS frente a AD y HP.",
            final_diagnosis="ASS",
        )

        assert isinstance(obj.reasoning, str)
        assert obj.reasoning

    def test_advanced_reasoning_dump_orders_reasoning_before_consensus(self):
        obj = AdvancedPolypClassificationWithReasoning(
            analysis_ad=ClassEvidence(evidence_for="a", evidence_against="b"),
            analysis_hp=ClassEvidence(evidence_for="a", evidence_against="b"),
            analysis_ass=ClassEvidence(evidence_for="a", evidence_against="b"),
            reasoning="observaciones y comparacion general de clases",
            clinical_consensus="consensus",
            final_diagnosis="ASS",
        )

        keys = list(obj.model_dump().keys())
        assert keys.index("reasoning") < keys.index("clinical_consensus")

    def test_reasoning_variant_requires_reasoning(self):
        with pytest.raises(ValidationError):
            PolypDetectionWithReasoning(
                polyp_detected=True,
                confidence_score=80,
                justification="Hallazgo compatible.",
            )

    def test_reasoning_variants_are_pydantic_models(self):
        for cls in (
            GenericObjectDetectionWithReasoning,
            PolypDetectionWithReasoning,
            PolypClassificationWithReasoning,
            AdvancedPolypClassificationWithReasoning,
            SycophancyTestWithReasoning,
            ImageQualityAssessmentWithReasoning,
        ):
            assert issubclass(cls, BaseModel)

    def test_reasoning_variant_preserves_base_field_metadata(self):
        for schema_name, base_cls in SCHEMA_REGISTRY.items():
            reasoning_cls = REASONING_SCHEMA_REGISTRY[schema_name]
            assert set(base_cls.model_fields.keys()).issubset(reasoning_cls.model_fields.keys())

            for field_name, base_field in base_cls.model_fields.items():
                reasoning_field = reasoning_cls.model_fields[field_name]
                assert base_field.annotation == reasoning_field.annotation
                assert base_field.description == reasoning_field.description
                assert base_field.metadata == reasoning_field.metadata

    def test_reasoning_variant_preserves_validators_and_config(self):
        class DemoSchema(BaseModel):
            model_config = ConfigDict(title="DemoSchemaTitle", extra="forbid")

            score: int = Field(ge=1, le=10, description="Synthetic score field.")

            @field_validator("score")
            @classmethod
            def reject_value_two(cls, value: int) -> int:
                if value == 2:
                    raise ValueError("score 2 is not allowed")
                return value

        DemoSchemaWithReasoning = _create_reasoning_schema(
            DemoSchema,
            reasoning_description="Synthetic reasoning.",
            docstring="Synthetic schema with reasoning.",
        )

        assert DemoSchemaWithReasoning.model_config.get("extra") == "forbid"
        assert DemoSchemaWithReasoning.model_json_schema().get("title") == "DemoSchemaTitle"

        with pytest.raises(ValidationError):
            DemoSchemaWithReasoning(reasoning="step by step", score=2)


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
        expected = {
            "GenericObjectDetection",
            "PolypDetection",
            "PolypClassification",
            "AdvancedPolypClassification",
            "SycophancyTest",
            "ImageQualityAssessment",
        }
        assert expected == set(SCHEMA_REGISTRY.keys())

    def test_values_are_pydantic_models(self):
        from pydantic import BaseModel
        for name, cls in SCHEMA_REGISTRY.items():
            assert issubclass(cls, BaseModel), f"{name} no es BaseModel"

    def test_registry_classes_match_imported(self):
        assert SCHEMA_REGISTRY["GenericObjectDetection"] is GenericObjectDetection
        assert SCHEMA_REGISTRY["PolypDetection"] is PolypDetection
        assert SCHEMA_REGISTRY["PolypClassification"] is PolypClassification
        assert SCHEMA_REGISTRY["AdvancedPolypClassification"] is AdvancedPolypClassification
        assert SCHEMA_REGISTRY["SycophancyTest"] is SycophancyTest
        assert SCHEMA_REGISTRY["ImageQualityAssessment"] is ImageQualityAssessment
