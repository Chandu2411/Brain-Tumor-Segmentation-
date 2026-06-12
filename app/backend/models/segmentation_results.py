from core.database import Base
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String


class Segmentation_results(Base):
    __tablename__ = "segmentation_results"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, nullable=False)
    user_id = Column(String, nullable=False)
    original_image_key = Column(String, nullable=False)
    segmented_image_key = Column(String, nullable=False)
    dice_coefficient = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    iou_score = Column(Float, nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.now)
    updated_at = Column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now)