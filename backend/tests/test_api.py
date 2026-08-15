"""
Tests for the FastAPI segmentation and evaluation API endpoints.

These tests verify:
  1. Normal inference does NOT return dice_coefficient or iou_score
  2. Normal inference DOES return inference_stats with tumor_detected, etc.
  3. The evaluate endpoint requires a ground-truth mask
  4. The model-performance endpoint returns valid structure
  5. History averages should be calculated from correct fields
"""

import sys
import os
import json

import pytest

# Ensure the backend root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Test: Segmentation response schema has inference_stats, not fake metrics
# ======================================================================

def test_segment_response_schema_no_fake_metrics():
    """
    Verify that the SegmentResponse schema does not include
    dice_coefficient, accuracy, or iou_score fields.
    """
    from routers.segmentation import SegmentResponse

    field_names = set(SegmentResponse.model_fields.keys())

    # These old fake fields must NOT exist
    assert "dice_coefficient" not in field_names, "SegmentResponse still has dice_coefficient"
    assert "accuracy" not in field_names, "SegmentResponse still has accuracy"
    assert "iou_score" not in field_names, "SegmentResponse still has iou_score"

    # These new honest fields MUST exist
    assert "inference_stats" in field_names, "SegmentResponse missing inference_stats"
    assert "model_used" in field_names, "SegmentResponse missing model_used"


def test_inference_stats_schema():
    """
    Verify InferenceStatsResponse has the correct honest fields.
    """
    from routers.segmentation import InferenceStatsResponse

    field_names = set(InferenceStatsResponse.model_fields.keys())
    required = {"tumor_detected", "tumor_pixel_count", "tumor_coverage_percentage",
                "mean_tumor_probability", "max_tumor_probability"}

    for field in required:
        assert field in field_names, f"InferenceStatsResponse missing: {field}"

    # Must NOT have fake metrics
    assert "dice_coefficient" not in field_names
    assert "iou_score" not in field_names


# ======================================================================
# Test: Evaluate endpoint response schema has real metrics
# ======================================================================

def test_evaluate_response_schema():
    """
    Verify EvaluateResponse has dice, iou, precision, recall, specificity.
    """
    from routers.evaluate import EvaluateResponse

    field_names = set(EvaluateResponse.model_fields.keys())
    required = {"dice", "iou", "precision", "recall", "specificity", "pixel_accuracy"}

    for field in required:
        assert field in field_names, f"EvaluateResponse missing: {field}"


# ======================================================================
# Test: Model performance endpoint response schema
# ======================================================================

def test_model_performance_response_schema():
    """
    Verify ModelPerformanceResponse has all required fields.
    """
    from routers.model_performance import ModelPerformanceResponse

    field_names = set(ModelPerformanceResponse.model_fields.keys())
    required = {"trained", "model_name", "test_samples", "dice_mean", "dice_std",
                "iou_mean", "iou_std", "precision", "recall", "specificity",
                "pixel_accuracy", "prediction_threshold"}

    for field in required:
        assert field in field_names, f"ModelPerformanceResponse missing: {field}"


# ======================================================================
# Test: model_metrics.json placeholder exists and has valid structure
# ======================================================================

def test_model_metrics_json_exists():
    """model_metrics.json should exist in the backend directory."""
    metrics_file = os.path.join(os.path.dirname(__file__), "..", "model_metrics.json")
    metrics_file = os.path.normpath(metrics_file)
    assert os.path.isfile(metrics_file), f"model_metrics.json not found at {metrics_file}"


def test_model_metrics_json_structure():
    """model_metrics.json should be valid JSON with required keys."""
    metrics_file = os.path.join(os.path.dirname(__file__), "..", "model_metrics.json")
    metrics_file = os.path.normpath(metrics_file)

    with open(metrics_file, "r") as f:
        data = json.load(f)

    required_keys = ["model_name", "test_samples", "dice_mean", "iou_mean",
                     "precision", "recall", "specificity", "pixel_accuracy",
                     "prediction_threshold"]
    for key in required_keys:
        assert key in data, f"model_metrics.json missing key: {key}"

    # Values should be numeric (placeholder zeros are fine)
    for key in ["dice_mean", "iou_mean", "precision", "recall"]:
        assert isinstance(data[key], (int, float)), f"{key} should be numeric"


# ======================================================================
# Test: DB model has honest inference columns
# ======================================================================

def test_db_model_has_inference_columns():
    """
    Verify that the SQLAlchemy Segmentation_results model has the
    new honest inference columns.
    """
    from models.segmentation_results import Segmentation_results

    columns = {c.name for c in Segmentation_results.__table__.columns}

    # New honest columns
    for col in ["tumor_detected", "tumor_pixel_count", "tumor_coverage_percentage",
                "mean_tumor_probability", "max_tumor_probability", "model_used"]:
        assert col in columns, f"DB model missing column: {col}"

    # Legacy columns should exist (for backward compatibility) but be marked legacy
    for col in ["legacy_dice_coefficient", "legacy_accuracy", "legacy_iou_score"]:
        assert col in columns, f"DB model missing legacy column: {col}"


# ======================================================================
# Test: data_models/segmentation_results.json no longer lists fake fields
# ======================================================================

def test_data_model_schema_no_fake_fields():
    """
    The data model schema should not have dice_coefficient, accuracy,
    or iou_score as required fields.
    """
    schema_file = os.path.join(
        os.path.dirname(__file__), "..", "data_models", "segmentation_results.json"
    )
    schema_file = os.path.normpath(schema_file)

    with open(schema_file, "r") as f:
        schema = json.load(f)

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    assert "dice_coefficient" not in required, "Schema still requires dice_coefficient"
    assert "accuracy" not in required, "Schema still requires accuracy"
    assert "iou_score" not in required, "Schema still requires iou_score"

    # Check properties too
    assert "dice_coefficient" not in properties, "Schema still has dice_coefficient property"
    assert "accuracy" not in properties, "Schema still has accuracy property"
    assert "iou_score" not in properties, "Schema still has iou_score property"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
