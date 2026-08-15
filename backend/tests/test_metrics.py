"""
Tests for the shared metric functions in metrics_utils.py

These tests verify:
  1. Dice and IoU return ~1 for perfect overlap
  2. Dice and IoU return ~0 for no overlap
  3. Dice-IoU consistency: IoU ≈ Dice / (2 - Dice)
  4. Precision, recall, specificity, pixel accuracy edge cases
  5. No hardcoded 0.78 / 0.72 values remain in the codebase
  6. Mask stays binary after nearest-neighbor resize
"""

import sys
import os
import re
import glob

import numpy as np
import pytest

# Ensure the backend root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics_utils import (
    binary_dice_np,
    binary_iou_np,
    pixel_precision_np,
    pixel_recall_np,
    pixel_specificity_np,
    pixel_accuracy_np,
    compute_all_metrics_np,
)


# ======================================================================
# Test 1 & 3: Dice perfect overlap ≈ 1
# ======================================================================

def test_dice_perfect_overlap():
    mask = np.ones((128, 128), dtype=np.float32)
    result = binary_dice_np(mask, mask)
    assert result > 0.999, f"Dice on perfect overlap should be ~1, got {result}"


# ======================================================================
# Test 2: Dice no overlap ≈ 0
# ======================================================================

def test_dice_no_overlap():
    y_true = np.ones((128, 128), dtype=np.float32)
    y_pred = np.zeros((128, 128), dtype=np.float32)
    result = binary_dice_np(y_true, y_pred)
    assert result < 0.001, f"Dice on no overlap should be ~0, got {result}"


# ======================================================================
# Test 3: IoU perfect overlap ≈ 1
# ======================================================================

def test_iou_perfect_overlap():
    mask = np.ones((128, 128), dtype=np.float32)
    result = binary_iou_np(mask, mask)
    assert result > 0.999, f"IoU on perfect overlap should be ~1, got {result}"


# ======================================================================
# Test 4: IoU no overlap ≈ 0
# ======================================================================

def test_iou_no_overlap():
    y_true = np.ones((128, 128), dtype=np.float32)
    y_pred = np.zeros((128, 128), dtype=np.float32)
    result = binary_iou_np(y_true, y_pred)
    assert result < 0.001, f"IoU on no overlap should be ~0, got {result}"


# ======================================================================
# Test 5: Dice-IoU consistency: IoU ≈ Dice / (2 - Dice)
# ======================================================================

def test_dice_iou_consistency():
    rng = np.random.RandomState(42)
    for _ in range(10):
        y_true = (rng.rand(128, 128) > 0.5).astype(np.float32)
        y_pred = (rng.rand(128, 128) > 0.5).astype(np.float32)

        dice = binary_dice_np(y_true, y_pred)
        iou = binary_iou_np(y_true, y_pred)

        if dice > 0.01:  # avoid division instability near zero
            expected_iou = dice / (2.0 - dice)
            assert abs(iou - expected_iou) < 0.01, (
                f"IoU ({iou:.6f}) should ≈ Dice/(2-Dice) = {expected_iou:.6f}"
            )


# ======================================================================
# Test 6: Precision on known values
# ======================================================================

def test_precision_known():
    # TP=4, FP=2, FN=0  →  precision = 4/(4+2) ≈ 0.667
    y_true = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
    y_pred = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
    prec = pixel_precision_np(y_true, y_pred)
    assert abs(prec - 4.0 / 6.0) < 0.01


# ======================================================================
# Test 7: Recall on known values
# ======================================================================

def test_recall_known():
    # TP=2, FN=2  →  recall = 2/4 = 0.5
    y_true = np.array([[1, 1], [1, 1]], dtype=np.float32)
    y_pred = np.array([[1, 0], [1, 0]], dtype=np.float32)
    rec = pixel_recall_np(y_true, y_pred)
    assert abs(rec - 0.5) < 0.01


# ======================================================================
# Test 8: Pixel accuracy on known values
# ======================================================================

def test_pixel_accuracy_known():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[1, 0], [0, 1]], dtype=np.float32)
    acc = pixel_accuracy_np(y_true, y_pred)
    assert acc == 1.0


# ======================================================================
# Test 9: compute_all_metrics_np returns correct keys
# ======================================================================

def test_compute_all_metrics_keys():
    y_true = np.ones((64, 64), dtype=np.float32)
    y_pred = np.ones((64, 64), dtype=np.float32) * 0.9
    metrics = compute_all_metrics_np(y_true, y_pred)
    for key in ("dice", "iou", "precision", "recall", "specificity", "pixel_accuracy"):
        assert key in metrics, f"Missing key: {key}"


# ======================================================================
# Test 10: Mask stays binary after nearest-neighbor resize (PIL)
# ======================================================================

def test_mask_binary_after_resize():
    from PIL import Image

    # Create a binary mask
    mask_np = np.zeros((256, 256), dtype=np.uint8)
    mask_np[50:200, 50:200] = 255

    img = Image.fromarray(mask_np, mode="L")
    resized = img.resize((128, 128), Image.NEAREST)
    arr = np.array(resized, dtype=np.float32) / 255.0
    binary = (arr >= 0.5).astype(np.float32)

    unique_vals = set(np.unique(binary).tolist())
    assert unique_vals.issubset({0.0, 1.0}), (
        f"Mask not binary after nearest-neighbor resize: {unique_vals}"
    )


# ======================================================================
# Test 11-16: No hardcoded 0.78/78/0.72/72 in codebase
# ======================================================================

def _scan_codebase_for_patterns(patterns: list) -> list:
    """Scan .py, .tsx, .ts files for suspicious hardcoded metric values."""
    backend_root = os.path.join(os.path.dirname(__file__), "..")
    frontend_root = os.path.normpath(os.path.join(backend_root, "..", "frontend", "src"))

    hits = []
    for root_dir in [backend_root, frontend_root]:
        for ext in ("*.py", "*.tsx", "*.ts"):
            for filepath in glob.glob(os.path.join(root_dir, "**", ext), recursive=True):
                # Skip this test file itself and __pycache__
                if "__pycache__" in filepath or "test_metrics" in filepath:
                    continue
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    for pattern in patterns:
                        if re.search(pattern, content):
                            hits.append((filepath, pattern))
                except Exception:
                    pass
    return hits


def test_no_hardcoded_078_dice():
    """No remaining hardcoded 0.78 Dice values in .py/.tsx/.ts files."""
    hits = _scan_codebase_for_patterns([
        r"dice.*=\s*0\.78",
        r"dice.*=\s*78\b",
        r"0\.78.*dice",
    ])
    assert len(hits) == 0, f"Found hardcoded 0.78/78 Dice references: {hits}"


def test_no_hardcoded_072_iou():
    """No remaining hardcoded 0.72 IoU values in .py/.tsx/.ts files."""
    hits = _scan_codebase_for_patterns([
        r"iou.*=\s*0\.72",
        r"iou.*=\s*72\b",
        r"0\.72.*iou",
    ])
    assert len(hits) == 0, f"Found hardcoded 0.72/72 IoU references: {hits}"


def test_no_fake_dice_in_frontend():
    """Frontend should not reference result.dice_coefficient or result.accuracy."""
    frontend_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src")
    )
    hits = []
    for ext in ("*.tsx", "*.ts"):
        for filepath in glob.glob(os.path.join(frontend_root, "**", ext), recursive=True):
            if "__pycache__" in filepath:
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Check if referencing the old fake field names as result properties
                if "result.dice_coefficient" in content or "r.dice_coefficient" in content:
                    hits.append((filepath, "dice_coefficient"))
                if "result.iou_score" in content or "r.iou_score" in content:
                    hits.append((filepath, "iou_score"))
                if "result.accuracy" in content or "r.accuracy" in content:
                    # Exclude legitimate usage (like pixel_accuracy)
                    if "pixel_accuracy" not in content.replace("result.accuracy", ""):
                        hits.append((filepath, "accuracy"))
            except Exception:
                pass
    assert len(hits) == 0, (
        f"Frontend still references old fake metric fields: {hits}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
