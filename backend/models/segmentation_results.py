from core.database import Base
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String


class Segmentation_results(Base):
    """
    Stores inference results for a single MRI upload.

    All fields are REAL inference statistics derived from the predicted mask.
    Dice and IoU are NOT stored here because they require a ground-truth mask,
    which a normal user does not supply.

    Legacy records created before this schema change will have NULL values in
    the new columns and are considered 'unverified / legacy'.
    """

    __tablename__ = "segmentation_results"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, nullable=False)
    user_id = Column(String, nullable=False, index=True)

    # Storage references (truncated data-URIs or object-store keys)
    original_image_key = Column(String, nullable=False)
    segmented_image_key = Column(String, nullable=False)

    # Original filename
    filename = Column(String, nullable=False)

    # -----------------------------------------------------------------------
    # Honest, prediction-derived inference statistics
    # -----------------------------------------------------------------------
    tumor_detected = Column(Boolean, nullable=True)
    """True if predicted tumor area >= MIN_TUMOR_PIXELS threshold."""

    tumor_pixel_count = Column(Integer, nullable=True)
    """Raw count of pixels above the 0.5 prediction threshold."""

    tumor_coverage_percentage = Column(Float, nullable=True)
    """tumor_pixel_count / total_pixels * 100  (0–100 %)."""

    mean_tumor_probability = Column(Float, nullable=True)
    """Mean sigmoid probability over the foreground (tumor) region."""

    max_tumor_probability = Column(Float, nullable=True)
    """Maximum sigmoid probability over the entire image."""

    # Model identification
    model_used = Column(String, nullable=True)
    """'Enhanced U-Net' or 'SciPy Fallback' — honest labelling of the model."""

    # -----------------------------------------------------------------------
    # Legacy columns — kept for backward-compatibility with old records.
    # Values in these columns for records created before this migration
    # are FABRICATED and must NOT be trusted.
    # -----------------------------------------------------------------------
    legacy_dice_coefficient = Column(Float, nullable=True)
    legacy_accuracy = Column(Float, nullable=True)
    legacy_iou_score = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.now)
    updated_at = Column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now)