import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.segmentation_results import Segmentation_results

logger = logging.getLogger(__name__)


class Segmentation_resultsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: dict, user_id: str) -> Segmentation_results:
        """Create a new segmentation result with honest inference statistics."""
        t0 = time.time()
        logger.debug("[DB_OP] Creating segmentation result for user_id=%s", user_id)

        db_obj = Segmentation_results(
            user_id=user_id,
            original_image_key=data.get("original_image_key", ""),
            segmented_image_key=data.get("segmented_image_key", ""),
            filename=data.get("filename", "unknown.png"),
            # Inference statistics
            tumor_detected=data.get("tumor_detected"),
            tumor_pixel_count=data.get("tumor_pixel_count"),
            tumor_coverage_percentage=data.get("tumor_coverage_percentage"),
            mean_tumor_probability=data.get("mean_tumor_probability"),
            max_tumor_probability=data.get("max_tumor_probability"),
            model_used=data.get("model_used"),
            # Legacy (never written for new records)
            legacy_dice_coefficient=None,
            legacy_accuracy=None,
            legacy_iou_score=None,
        )
        self.db.add(db_obj)
        await self.db.commit()
        await self.db.refresh(db_obj)
        logger.debug("[DB_OP] Create completed in %.4fs", time.time() - t0)
        return db_obj

    async def get_by_id(
        self, id: int, user_id: Optional[str] = None
    ) -> Optional[Segmentation_results]:
        """Get a single segmentation result by ID."""
        t0 = time.time()
        stmt = select(Segmentation_results).where(Segmentation_results.id == id)
        if user_id is not None:
            stmt = stmt.where(Segmentation_results.user_id == user_id)
        result = await self.db.execute(stmt)
        db_obj = result.scalar_one_or_none()
        logger.debug("[DB_OP] get_by_id completed in %.4fs", time.time() - t0)
        return db_obj

    async def get_list(
        self,
        skip: int = 0,
        limit: int = 20,
        query_dict: Optional[dict] = None,
        sort: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        """Get paginated, sorted segmentation results."""
        t0 = time.time()
        stmt = select(Segmentation_results)
        if user_id is not None:
            stmt = stmt.where(Segmentation_results.user_id == user_id)

        if query_dict:
            for key, value in query_dict.items():
                if hasattr(Segmentation_results, key):
                    stmt = stmt.where(getattr(Segmentation_results, key) == value)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.db.execute(count_stmt)).scalar_one()

        if sort:
            field_name = sort.lstrip("-")
            if hasattr(Segmentation_results, field_name):
                col = getattr(Segmentation_results, field_name)
                stmt = stmt.order_by(desc(col) if sort.startswith("-") else asc(col))
        else:
            stmt = stmt.order_by(desc(Segmentation_results.created_at))

        stmt = stmt.offset(skip).limit(limit)
        items = (await self.db.execute(stmt)).scalars().all()

        logger.debug("[DB_OP] get_list: %d items in %.4fs", len(items), time.time() - t0)
        return {"items": list(items), "total": total, "skip": skip, "limit": limit}

    async def update(
        self, id: int, data: dict, user_id: Optional[str] = None
    ) -> Optional[Segmentation_results]:
        """Partially update a segmentation result."""
        db_obj = await self.get_by_id(id, user_id=user_id)
        if not db_obj:
            return None
        for key, value in data.items():
            if hasattr(db_obj, key):
                setattr(db_obj, key, value)
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def delete(self, id: int, user_id: Optional[str] = None) -> bool:
        """Delete a segmentation result."""
        db_obj = await self.get_by_id(id, user_id=user_id)
        if not db_obj:
            return False
        await self.db.delete(db_obj)
        await self.db.commit()
        return True
