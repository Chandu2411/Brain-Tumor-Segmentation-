import logging
import base64
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
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
    image: str  # base64 data URI
    filename: Optional[str] = "unknown.png"


class SegmentResponse(BaseModel):
    success: bool
    overlay_image: Optional[str] = None
    mask_image: Optional[str] = None
    dice_coefficient: Optional[float] = None
    accuracy: Optional[float] = None
    iou_score: Optional[float] = None
    result_id: Optional[int] = None
    error: Optional[str] = None


@router.post("/analyze", response_model=SegmentResponse)
async def analyze_mri(
    data: SegmentRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Analyze an MRI scan for brain tumor segmentation"""
    try:
        # Decode base64 image
        if "," in data.image:
            image_data = base64.b64decode(data.image.split(",")[1])
        else:
            image_data = base64.b64decode(data.image)

        # Run segmentation
        service = SegmentationService()
        result = await service.segment_image(image_data)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Segmentation failed"))

        # Save result to database
        results_service = Segmentation_resultsService(db)
        db_result = await results_service.create(
            data={
                "original_image_key": data.image[:100] + "...",  # Store truncated reference
                "segmented_image_key": result["overlay_image"][:100] + "...",
                "dice_coefficient": result["metrics"]["dice_coefficient"],
                "accuracy": result["metrics"]["accuracy"],
                "iou_score": result["metrics"]["iou_score"],
                "filename": data.filename or "unknown.png",
            },
            user_id=current_user.id,
        )

        return SegmentResponse(
            success=True,
            overlay_image=result["overlay_image"],
            mask_image=result["mask_image"],
            dice_coefficient=result["metrics"]["dice_coefficient"],
            accuracy=result["metrics"]["accuracy"],
            iou_score=result["metrics"]["iou_score"],
            result_id=db_result.id if db_result else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/history")
async def get_history(
    skip: int = 0,
    limit: int = 20,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get segmentation history for the current user"""
    try:
        results_service = Segmentation_resultsService(db)
        results = await results_service.get_list(
            skip=skip,
            limit=limit,
            user_id=current_user.id,
            sort="-created_at",
        )
        return results
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")