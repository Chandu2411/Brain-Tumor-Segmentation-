import json
import logging
from typing import List, Optional

from datetime import datetime, date

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from services.segmentation_results import Segmentation_resultsService
from dependencies.auth import get_current_user
from schemas.auth import UserResponse

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/entities/segmentation_results", tags=["segmentation_results"])


# ---------- Pydantic Schemas ----------
class Segmentation_resultsData(BaseModel):
    """Entity data schema (for create/update)"""
    original_image_key: str
    segmented_image_key: str
    dice_coefficient: float
    accuracy: float
    iou_score: float
    filename: str


class Segmentation_resultsUpdateData(BaseModel):
    """Update entity data (partial updates allowed)"""
    original_image_key: Optional[str] = None
    segmented_image_key: Optional[str] = None
    dice_coefficient: Optional[float] = None
    accuracy: Optional[float] = None
    iou_score: Optional[float] = None
    filename: Optional[str] = None


class Segmentation_resultsResponse(BaseModel):
    """Entity response schema"""
    id: int
    user_id: str
    original_image_key: str
    segmented_image_key: str
    dice_coefficient: float
    accuracy: float
    iou_score: float
    filename: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Segmentation_resultsListResponse(BaseModel):
    """List response schema"""
    items: List[Segmentation_resultsResponse]
    total: int
    skip: int
    limit: int


class Segmentation_resultsBatchCreateRequest(BaseModel):
    """Batch create request"""
    items: List[Segmentation_resultsData]


class Segmentation_resultsBatchUpdateItem(BaseModel):
    """Batch update item"""
    id: int
    updates: Segmentation_resultsUpdateData


class Segmentation_resultsBatchUpdateRequest(BaseModel):
    """Batch update request"""
    items: List[Segmentation_resultsBatchUpdateItem]


class Segmentation_resultsBatchDeleteRequest(BaseModel):
    """Batch delete request"""
    ids: List[int]


# ---------- Routes ----------
@router.get("", response_model=Segmentation_resultsListResponse)
async def query_segmentation_resultss(
    query: str = Query(None, description="Query conditions (JSON string)"),
    sort: str = Query(None, description="Sort field (prefix with '-' for descending)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=2000, description="Max number of records to return"),
    fields: str = Query(None, description="Comma-separated list of fields to return"),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Query segmentation_resultss with filtering, sorting, and pagination (user can only see their own records)"""
    logger.debug(f"Querying segmentation_resultss: query={query}, sort={sort}, skip={skip}, limit={limit}, fields={fields}")
    
    service = Segmentation_resultsService(db)
    try:
        # Parse query JSON if provided
        query_dict = None
        if query:
            try:
                query_dict = json.loads(query)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid query JSON format")
        
        result = await service.get_list(
            skip=skip, 
            limit=limit,
            query_dict=query_dict,
            sort=sort,
            user_id=str(current_user.id),
        )
        logger.debug(f"Found {result['total']} segmentation_resultss")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying segmentation_resultss: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/all", response_model=Segmentation_resultsListResponse)
async def query_segmentation_resultss_all(
    query: str = Query(None, description="Query conditions (JSON string)"),
    sort: str = Query(None, description="Sort field (prefix with '-' for descending)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=2000, description="Max number of records to return"),
    fields: str = Query(None, description="Comma-separated list of fields to return"),
    db: AsyncSession = Depends(get_db),
):
    # Query segmentation_resultss with filtering, sorting, and pagination without user limitation
    logger.debug(f"Querying segmentation_resultss: query={query}, sort={sort}, skip={skip}, limit={limit}, fields={fields}")

    service = Segmentation_resultsService(db)
    try:
        # Parse query JSON if provided
        query_dict = None
        if query:
            try:
                query_dict = json.loads(query)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid query JSON format")

        result = await service.get_list(
            skip=skip,
            limit=limit,
            query_dict=query_dict,
            sort=sort
        )
        logger.debug(f"Found {result['total']} segmentation_resultss")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying segmentation_resultss: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{id}", response_model=Segmentation_resultsResponse)
async def get_segmentation_results(
    id: int,
    fields: str = Query(None, description="Comma-separated list of fields to return"),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single segmentation_results by ID (user can only see their own records)"""
    logger.debug(f"Fetching segmentation_results with id: {id}, fields={fields}")
    
    service = Segmentation_resultsService(db)
    try:
        result = await service.get_by_id(id, user_id=str(current_user.id))
        if not result:
            logger.warning(f"Segmentation_results with id {id} not found")
            raise HTTPException(status_code=404, detail="Segmentation_results not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching segmentation_results {id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("", response_model=Segmentation_resultsResponse, status_code=201)
async def create_segmentation_results(
    data: Segmentation_resultsData,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new segmentation_results"""
    logger.debug(f"Creating new segmentation_results with data: {data}")
    
    service = Segmentation_resultsService(db)
    try:
        result = await service.create(data.model_dump(), user_id=str(current_user.id))
        if not result:
            raise HTTPException(status_code=400, detail="Failed to create segmentation_results")
        
        logger.info(f"Segmentation_results created successfully with id: {result.id}")
        return result
    except ValueError as e:
        logger.error(f"Validation error creating segmentation_results: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating segmentation_results: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch", response_model=List[Segmentation_resultsResponse], status_code=201)
async def create_segmentation_resultss_batch(
    request: Segmentation_resultsBatchCreateRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create multiple segmentation_resultss in a single request"""
    logger.debug(f"Batch creating {len(request.items)} segmentation_resultss")
    
    service = Segmentation_resultsService(db)
    results = []
    
    try:
        for item_data in request.items:
            result = await service.create(item_data.model_dump(), user_id=str(current_user.id))
            if result:
                results.append(result)
        
        logger.info(f"Batch created {len(results)} segmentation_resultss successfully")
        return results
    except Exception as e:
        await db.rollback()
        logger.error(f"Error in batch create: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch create failed: {str(e)}")


@router.put("/batch", response_model=List[Segmentation_resultsResponse])
async def update_segmentation_resultss_batch(
    request: Segmentation_resultsBatchUpdateRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update multiple segmentation_resultss in a single request (requires ownership)"""
    logger.debug(f"Batch updating {len(request.items)} segmentation_resultss")
    
    service = Segmentation_resultsService(db)
    results = []
    
    try:
        for item in request.items:
            # Only include non-None values for partial updates
            update_dict = {k: v for k, v in item.updates.model_dump().items() if v is not None}
            result = await service.update(item.id, update_dict, user_id=str(current_user.id))
            if result:
                results.append(result)
        
        logger.info(f"Batch updated {len(results)} segmentation_resultss successfully")
        return results
    except Exception as e:
        await db.rollback()
        logger.error(f"Error in batch update: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch update failed: {str(e)}")


@router.put("/{id}", response_model=Segmentation_resultsResponse)
async def update_segmentation_results(
    id: int,
    data: Segmentation_resultsUpdateData,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing segmentation_results (requires ownership)"""
    logger.debug(f"Updating segmentation_results {id} with data: {data}")

    service = Segmentation_resultsService(db)
    try:
        # Only include non-None values for partial updates
        update_dict = {k: v for k, v in data.model_dump().items() if v is not None}
        result = await service.update(id, update_dict, user_id=str(current_user.id))
        if not result:
            logger.warning(f"Segmentation_results with id {id} not found for update")
            raise HTTPException(status_code=404, detail="Segmentation_results not found")
        
        logger.info(f"Segmentation_results {id} updated successfully")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating segmentation_results {id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating segmentation_results {id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/batch")
async def delete_segmentation_resultss_batch(
    request: Segmentation_resultsBatchDeleteRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple segmentation_resultss by their IDs (requires ownership)"""
    logger.debug(f"Batch deleting {len(request.ids)} segmentation_resultss")
    
    service = Segmentation_resultsService(db)
    deleted_count = 0
    
    try:
        for item_id in request.ids:
            success = await service.delete(item_id, user_id=str(current_user.id))
            if success:
                deleted_count += 1
        
        logger.info(f"Batch deleted {deleted_count} segmentation_resultss successfully")
        return {"message": f"Successfully deleted {deleted_count} segmentation_resultss", "deleted_count": deleted_count}
    except Exception as e:
        await db.rollback()
        logger.error(f"Error in batch delete: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch delete failed: {str(e)}")


@router.delete("/{id}")
async def delete_segmentation_results(
    id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a single segmentation_results by ID (requires ownership)"""
    logger.debug(f"Deleting segmentation_results with id: {id}")
    
    service = Segmentation_resultsService(db)
    try:
        success = await service.delete(id, user_id=str(current_user.id))
        if not success:
            logger.warning(f"Segmentation_results with id {id} not found for deletion")
            raise HTTPException(status_code=404, detail="Segmentation_results not found")
        
        logger.info(f"Segmentation_results {id} deleted successfully")
        return {"message": "Segmentation_results deleted successfully", "id": id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting segmentation_results {id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")