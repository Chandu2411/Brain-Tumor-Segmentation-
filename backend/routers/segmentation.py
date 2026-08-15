import logging
import base64
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from core.database import get_db
from dependencies.auth import get_current_user
from schemas.auth import UserResponse
from services.segmentation import SegmentationService
from services.segmentation_results import Segmentation_resultsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/segmentation", tags=["segmentation"])


class SegmentRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded data-URI of the MRI scan")
    filename: Optional[str] = Field("unknown.png", description="Original filename")


class InferenceStatsResponse(BaseModel):
    """
    Honest, prediction-derived statistics for a plain MRI upload.

    Note: Dice and IoU are intentionally absent. They require a ground-truth
    mask, which is not supplied during normal inference. Use POST /api/evaluate
    to compute Dice / IoU when you have both MRI and ground-truth mask.
    """
    tumor_detected: bool = Field(..., description="True if predicted tumor area >= threshold")
    tumor_pixel_count: int = Field(..., description="Number of pixels above 0.5 threshold")
    tumor_coverage_percentage: float = Field(..., description="Tumor pixels / total pixels × 100")
    mean_tumor_probability: float = Field(
        ..., description="Mean sigmoid probability over foreground region"
    )
    max_tumor_probability: float = Field(
        ..., description="Maximum sigmoid probability over entire image"
    )


class SegmentResponse(BaseModel):
    success: bool
    overlay_image: Optional[str] = None
    mask_image: Optional[str] = None
    inference_stats: Optional[InferenceStatsResponse] = None
    model_used: Optional[str] = None
    result_id: Optional[int] = None
    error: Optional[str] = None


@router.post("/analyze", response_model=SegmentResponse)
async def analyze_mri(
    data: SegmentRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze an MRI scan for brain tumor segmentation.

    Returns honest inference statistics — never fabricated Dice or IoU.
    The 'inference_stats' field contains:
      - tumor_detected
      - tumor_pixel_count
      - tumor_coverage_percentage   (tumor pixels / total pixels × 100)
      - mean_tumor_probability
      - max_tumor_probability

    These values reflect actual model predictions, not ground-truth comparisons.
    For Dice/IoU evaluation, use POST /api/evaluate with a ground-truth mask.
    """
    try:
        # Decode base64 image
        if "," in data.image:
            image_bytes = base64.b64decode(data.image.split(",")[1])
        else:
            image_bytes = base64.b64decode(data.image)

        # Run segmentation
        service = SegmentationService()
        result = await service.segment_image(image_bytes)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Segmentation failed")
            )

        stats = result["inference_stats"]

        # Persist result to database
        results_service = Segmentation_resultsService(db)
        db_result = await results_service.create(
            data={
                "original_image_key": data.image[:100] + "...",
                "segmented_image_key": result["overlay_image"][:100] + "...",
                "filename": data.filename or "unknown.png",
                "tumor_detected": stats["tumor_detected"],
                "tumor_pixel_count": stats["tumor_pixel_count"],
                "tumor_coverage_percentage": stats["tumor_coverage_percentage"],
                "mean_tumor_probability": stats["mean_tumor_probability"],
                "max_tumor_probability": stats["max_tumor_probability"],
                "model_used": result.get("model_used", "Unknown"),
            },
            user_id=str(current_user.id),
        )

        return SegmentResponse(
            success=True,
            overlay_image=result["overlay_image"],
            mask_image=result["mask_image"],
            inference_stats=InferenceStatsResponse(**stats),
            model_used=result.get("model_used"),
            result_id=db_result.id if db_result else None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Segmentation analysis error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


@router.get("/history")
async def get_history(
    skip: int = 0,
    limit: int = 20,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get segmentation history for the current user (honest inference statistics only)."""
    try:
        results_service = Segmentation_resultsService(db)
        results = await results_service.get_list(
            skip=skip,
            limit=limit,
            user_id=str(current_user.id),
            sort="-created_at",
        )
        return results
    except Exception as exc:
        logger.error("History retrieval error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(exc)}")