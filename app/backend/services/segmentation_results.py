import logging
import time
from typing import Dict, Any, List, Optional
from sqlalchemy import select, desc, asc, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.segmentation_results import Segmentation_results

logger = logging.getLogger(__name__)


class Segmentation_resultsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: dict, user_id: str) -> Segmentation_results:
        """Create a new segmentation result."""
        start_time = time.time()
        logger.debug(f"[DB_OP] Starting create segmentation result - user_id: {user_id}")

        db_obj = Segmentation_results(
            user_id=user_id,
            original_image_key=data.get("original_image_key"),
            segmented_image_key=data.get("segmented_image_key"),
            dice_coefficient=data.get("dice_coefficient"),
            accuracy=data.get("accuracy"),
            iou_score=data.get("iou_score"),
            filename=data.get("filename"),
        )
        self.db.add(db_obj)
        await self.db.commit()
        await self.db.refresh(db_obj)

        logger.debug(f"[DB_OP] Create segmentation result completed in {time.time() - start_time:.4f}s")
        return db_obj

    async def get_by_id(self, id: int, user_id: Optional[str] = None) -> Optional[Segmentation_results]:
        """Get a single segmentation result by ID."""
        start_time = time.time()
        logger.debug(f"[DB_OP] Starting get_by_id - id: {id}, user_id: {user_id}")

        stmt = select(Segmentation_results).where(Segmentation_results.id == id)
        if user_id is not None:
            stmt = stmt.where(Segmentation_results.user_id == user_id)

        result = await self.db.execute(stmt)
        db_obj = result.scalar_one_or_none()

        logger.debug(f"[DB_OP] Get segmentation result completed in {time.time() - start_time:.4f}s")
        return db_obj

    async def get_list(
        self,
        skip: int = 0,
        limit: int = 20,
        query_dict: Optional[dict] = None,
        sort: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        """Get a list of segmentation results with filtering, sorting, and pagination."""
        start_time = time.time()
        logger.debug(f"[DB_OP] Starting get_list - skip: {skip}, limit: {limit}, sort: {sort}, user_id: {user_id}")

        # Build query
        stmt = select(Segmentation_results)
        if user_id is not None:
            stmt = stmt.where(Segmentation_results.user_id == user_id)

        if query_dict:
            for key, value in query_dict.items():
                if hasattr(Segmentation_results, key):
                    stmt = stmt.where(getattr(Segmentation_results, key) == value)

        # Count total
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar_one()

        # Sort
        if sort:
            if sort.startswith("-"):
                field_name = sort[1:]
                if hasattr(Segmentation_results, field_name):
                    stmt = stmt.order_by(desc(getattr(Segmentation_results, field_name)))
            else:
                field_name = sort
                if hasattr(Segmentation_results, field_name):
                    stmt = stmt.order_by(asc(getattr(Segmentation_results, field_name)))
        else:
            stmt = stmt.order_by(desc(Segmentation_results.created_at))

        # Pagination
        stmt = stmt.offset(skip).limit(limit)

        result = await self.db.execute(stmt)
        items = result.scalars().all()

        logger.debug(f"[DB_OP] Get list completed in {time.time() - start_time:.4f}s - found: {len(items)}")

        return {
            "items": list(items),
            "total": total,
            "skip": skip,
            "limit": limit
        }

    async def update(self, id: int, data: dict, user_id: Optional[str] = None) -> Optional[Segmentation_results]:
        """Update a segmentation result."""
        start_time = time.time()
        logger.debug(f"[DB_OP] Starting update - id: {id}, user_id: {user_id}")

        db_obj = await self.get_by_id(id, user_id=user_id)
        if not db_obj:
            return None

        for key, value in data.items():
            if hasattr(db_obj, key):
                setattr(db_obj, key, value)

        await self.db.commit()
        await self.db.refresh(db_obj)

        logger.debug(f"[DB_OP] Update completed in {time.time() - start_time:.4f}s")
        return db_obj

    async def delete(self, id: int, user_id: Optional[str] = None) -> bool:
        """Delete a segmentation result."""
        start_time = time.time()
        logger.debug(f"[DB_OP] Starting delete - id: {id}, user_id: {user_id}")

        db_obj = await self.get_by_id(id, user_id=user_id)
        if not db_obj:
            return False

        await self.db.delete(db_obj)
        await self.db.commit()

        logger.debug(f"[DB_OP] Delete completed in {time.time() - start_time:.4f}s")
        return True
