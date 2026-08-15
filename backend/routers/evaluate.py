"""
Evaluation router — POST /api/evaluate

This endpoint computes REAL supervised segmentation metrics when BOTH
an MRI image AND a ground-truth mask are supplied.

Normal inference (POST /api/v1/segmentation/analyze) does NOT have a
ground-truth mask, so Dice / IoU cannot be calculated there.  This
endpoint exists specifically for evaluation purposes.
"""

import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from services.segmentation import (
    INPUT_SIZE,
    PREDICTION_THRESHOLD,
    _try_load_keras_model,
    _scipy_segment,
    preprocess_image_for_inference,
    preprocess_mask_for_evaluation,
)
from metrics_utils import compute_all_metrics_np

import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["evaluate"])


class EvaluateRequest(BaseModel):
    """Request body for the evaluation endpoint."""
    image: str = Field(..., description="Base64-encoded data-URI of the MRI scan")
    mask: str = Field(..., description="Base64-encoded data-URI of the ground-truth mask")
    filename: Optional[str] = Field("unknown.png", description="Original filename")


class EvaluateResponse(BaseModel):
    """Response with real supervised metrics."""
    success: bool
    model_used: str
    filename: str
    dice: float = Field(..., description="Binary Dice coefficient (threshold=0.5)")
    iou: float = Field(..., description="Binary IoU / Jaccard index (threshold=0.5)")
    precision: float = Field(..., description="Pixel-level precision = TP/(TP+FP)")
    recall: float = Field(..., description="Pixel-level recall/sensitivity = TP/(TP+FN)")
    specificity: float = Field(..., description="Pixel-level specificity = TN/(TN+FP)")
    pixel_accuracy: float = Field(..., description="Pixel accuracy = (TP+TN)/total")
    prediction_threshold: float = PREDICTION_THRESHOLD
    error: Optional[str] = None


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_with_ground_truth(data: EvaluateRequest):
    """
    Evaluate segmentation quality by comparing the model prediction
    against a supplied ground-truth mask.

    Both the MRI image and the ground-truth mask must be provided as
    base64-encoded data URIs (or plain base64).

    Returns real Dice, IoU, precision, recall, specificity, pixel accuracy.
    """
    try:
        # --- Decode MRI image ---
        if "," in data.image:
            image_bytes = base64.b64decode(data.image.split(",")[1])
        else:
            image_bytes = base64.b64decode(data.image)

        # --- Decode ground-truth mask ---
        if "," in data.mask:
            mask_bytes = base64.b64decode(data.mask.split(",")[1])
        else:
            mask_bytes = base64.b64decode(data.mask)

        mri_image = Image.open(io.BytesIO(image_bytes))
        mask_image = Image.open(io.BytesIO(mask_bytes))

        # --- Preprocess ---
        img_batch = preprocess_image_for_inference(mri_image)   # (1, H, W, 3)
        gt_mask = preprocess_mask_for_evaluation(mask_image)     # (H, W) binary

        # --- Run model ---
        model = _try_load_keras_model()
        if model is not None:
            raw_pred = model.predict(img_batch, verbose=0)       # (1, H, W, 1)
            prob_mask = raw_pred[0, :, :, 0]                     # (H, W)
            model_used = "Enhanced U-Net"
        else:
            img_rgb = np.array(
                mri_image.convert("RGB").resize(
                    (INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            prob_mask = _scipy_segment(img_rgb)
            model_used = "SciPy Fallback (no trained model loaded)"

        # --- Compute real metrics ---
        metrics = compute_all_metrics_np(gt_mask, prob_mask, threshold=PREDICTION_THRESHOLD)

        return EvaluateResponse(
            success=True,
            model_used=model_used,
            filename=data.filename or "unknown.png",
            dice=metrics["dice"],
            iou=metrics["iou"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            specificity=metrics["specificity"],
            pixel_accuracy=metrics["pixel_accuracy"],
            prediction_threshold=PREDICTION_THRESHOLD,
        )

    except Exception as exc:
        logger.error("Evaluation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(exc)}")
