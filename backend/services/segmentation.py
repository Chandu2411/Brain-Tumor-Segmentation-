"""
Brain Tumor Segmentation Service.

This module performs MRI segmentation and computes HONEST, prediction-derived
inference statistics. Dice and IoU CANNOT be computed without a ground-truth
mask, so they are never fabricated here.

Metrics produced for a plain MRI upload (no ground truth):
    tumor_detected              -- bool: True if predicted tumor area >= MIN_TUMOR_PIXELS
    tumor_pixel_count           -- int:  number of pixels classified as tumor
    tumor_coverage_percentage   -- float: tumor_pixels / total_pixels * 100
    mean_tumor_probability      -- float: mean sigmoid probability in foreground region
    max_tumor_probability       -- float: max sigmoid probability over entire image

Real Dice / IoU / precision / recall require a ground-truth mask and are
computed only in the /api/evaluate endpoint (routers/evaluate.py).
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: Tuple[int, int] = (128, 128)          # (height, width)
PREDICTION_THRESHOLD: float = 0.5                  # binarisation threshold
MIN_TUMOR_PIXELS: int = 50                         # minimum pixels to call "detected"

# ---------------------------------------------------------------------------
# Optional real-model loading
# ---------------------------------------------------------------------------
_tf_model = None
_model_load_attempted = False


def _try_load_keras_model() -> Optional[Any]:
    """
    Attempt to load best_enhanced_unet.keras from the backend directory.
    Called once at first inference. Returns the model or None.
    """
    global _tf_model, _model_load_attempted
    if _model_load_attempted:
        return _tf_model
    _model_load_attempted = True

    import os
    model_path = os.path.join(os.path.dirname(__file__), "..", "best_enhanced_unet.keras")
    model_path = os.path.normpath(model_path)

    if not os.path.isfile(model_path):
        logger.info(
            "best_enhanced_unet.keras not found at %s — "
            "using fallback SciPy segmentation. Place the trained model there "
            "to enable real Enhanced U-Net inference.",
            model_path,
        )
        return None

    try:
        import tensorflow as tf  # noqa: F401 — only imported when model exists
        import sys as _sys
        _backend_dir = os.path.dirname(os.path.dirname(__file__))
        if _backend_dir not in _sys.path:
            _sys.path.insert(0, _backend_dir)
        from metrics_utils import bce_dice_loss, dice_coef, iou_coef  # type: ignore

        _tf_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "dice_coef": dice_coef,
                "iou_coef": iou_coef,
            },
        )
        logger.info(
            "Loaded Enhanced U-Net from %s | params=%d | input=%s | output=%s",
            model_path,
            _tf_model.count_params(),
            _tf_model.input_shape,
            _tf_model.output_shape,
        )
    except Exception as exc:
        logger.error("Failed to load Enhanced U-Net model: %s", exc)
        _tf_model = None

    return _tf_model


# ---------------------------------------------------------------------------
# Preprocessing (shared between inference and evaluation)
# ---------------------------------------------------------------------------

def preprocess_image_for_inference(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image for Enhanced U-Net inference.

    Pipeline (matches training preprocessing):
        1. Convert to RGB
        2. Resize to INPUT_SIZE with bilinear interpolation
        3. Cast to float32
        4. Normalize to [0, 1]

    Returns
    -------
    np.ndarray  shape (1, H, W, 3)  dtype float32
    """
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
    arr = np.array(image_resized, dtype=np.float32) / 255.0
    assert arr.shape == (INPUT_SIZE[0], INPUT_SIZE[1], 3), \
        f"Unexpected image shape after preprocessing: {arr.shape}"
    assert not np.any(np.isnan(arr)), "NaN values in preprocessed image"
    assert arr.min() >= 0.0 and arr.max() <= 1.0, \
        f"Image not in [0,1]: min={arr.min()}, max={arr.max()}"
    return arr[np.newaxis, ...]  # (1, H, W, 3)


def preprocess_mask_for_evaluation(mask_image: Image.Image) -> np.ndarray:
    """
    Preprocess a ground-truth mask for evaluation.

    Pipeline:
        1. Convert to grayscale
        2. Resize to INPUT_SIZE with nearest-neighbour interpolation (no interpolation artifacts)
        3. Binarise at threshold 0.5 (after normalising to [0,1])
        4. Cast to float32, shape (H, W)

    Returns
    -------
    np.ndarray  shape (H, W)  dtype float32  values ∈ {0.0, 1.0}
    """
    mask_gray = mask_image.convert("L")
    mask_resized = mask_gray.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.NEAREST)
    arr = np.array(mask_resized, dtype=np.float32) / 255.0
    binary = (arr >= 0.5).astype(np.float32)
    unique = np.unique(binary)
    assert set(unique.tolist()).issubset({0.0, 1.0}), \
        f"Mask values not binary after binarisation: {unique}"
    return binary


# ---------------------------------------------------------------------------
# Fallback SciPy segmentation (used when no .keras model is available)
# ---------------------------------------------------------------------------

def _scipy_segment(img_array: np.ndarray) -> np.ndarray:
    """
    Fallback segmentation using classical image processing.

    This is NOT the Enhanced U-Net. It is clearly labelled as a fallback
    and produces ONLY a visual mask — no performance metrics.

    Parameters
    ----------
    img_array : np.ndarray  shape (H, W, 3) or (H, W)  float32 in [0, 1]

    Returns
    -------
    np.ndarray  shape (H, W)  float32  values in [0, 1] (raw probability proxy)
    """
    from scipy import ndimage  # imported here to avoid hard dependency at module level

    # Work on grayscale
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=-1)
    else:
        gray = img_array.astype(np.float32)

    h, w = gray.shape

    # Multi-scale smoothing
    smooth_1 = ndimage.gaussian_filter(gray, sigma=1.0)
    smooth_2 = ndimage.gaussian_filter(gray, sigma=2.0)

    mean_val = np.mean(gray)
    std_val = np.std(gray)

    # Intensity-based thresholding
    threshold_high = mean_val + 1.2 * std_val
    threshold_low = mean_val + 0.8 * std_val

    binary_high = (gray > threshold_high).astype(np.float32)
    binary_low = (gray > threshold_low).astype(np.float32)

    # Edge features
    edges_1 = ndimage.sobel(smooth_1)
    edges_2 = ndimage.sobel(smooth_2)
    edge_combined = edges_1 * 0.3 + edges_2 * 0.7
    edge_norm = edge_combined / (edge_combined.max() + 1e-7)

    # Feature fusion
    fused = binary_high * 0.6 + (binary_low * (1.0 - edge_norm * 0.5)) * 0.4

    # Morphological cleanup
    struct = ndimage.generate_binary_structure(2, 2)
    mask = ndimage.binary_opening(fused > 0.4, structure=struct, iterations=2)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=3)

    # Keep largest connected component
    labeled, n_features = ndimage.label(mask)
    if n_features > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = (labeled == largest).astype(np.float32)
        # Slight boundary smoothing (for visual quality only)
        mask = ndimage.gaussian_filter(mask, sigma=1.5)
    else:
        # No region found — return empty mask
        mask = np.zeros((h, w), dtype=np.float32)

    return mask  # values in [0, 1]


# ---------------------------------------------------------------------------
# Inference statistics (no ground truth required)
# ---------------------------------------------------------------------------

def compute_inference_stats(prob_mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute honest inference statistics from a predicted probability mask.

    Parameters
    ----------
    prob_mask : np.ndarray  shape (H, W)  float32 in [0, 1]
        Raw sigmoid output (or SciPy proxy) — NOT binarised yet.

    Returns
    -------
    dict with keys:
        tumor_detected              bool
        tumor_pixel_count           int
        tumor_coverage_percentage   float  (0–100)
        mean_tumor_probability      float  (0–1)
        max_tumor_probability       float  (0–1)
    """
    assert prob_mask.ndim == 2, f"Expected 2-D mask, got shape {prob_mask.shape}"
    assert not np.any(np.isnan(prob_mask)), "NaN values in probability mask"

    total_pixels = int(prob_mask.size)
    binary_mask = (prob_mask >= PREDICTION_THRESHOLD).astype(np.float32)
    tumor_pixel_count = int(np.sum(binary_mask))

    tumor_detected = tumor_pixel_count >= MIN_TUMOR_PIXELS
    tumor_coverage_pct = round(float(tumor_pixel_count / total_pixels * 100), 4)
    max_prob = round(float(np.max(prob_mask)), 4)

    # Mean probability only over the foreground region; 0 if no foreground
    if tumor_pixel_count > 0:
        mean_prob = round(float(np.sum(prob_mask * binary_mask) / tumor_pixel_count), 4)
    else:
        mean_prob = 0.0

    return {
        "tumor_detected": tumor_detected,
        "tumor_pixel_count": tumor_pixel_count,
        "tumor_coverage_percentage": tumor_coverage_pct,
        "mean_tumor_probability": mean_prob,
        "max_tumor_probability": max_prob,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def create_overlay_image(
    original: Image.Image,
    prob_mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 100),
    opacity: float = 0.5,
) -> Image.Image:
    """Overlay the binary mask on the original image with a coloured highlight."""
    from scipy import ndimage  # local import

    original_resized = original.convert("RGB").resize(
        (INPUT_SIZE[1], INPUT_SIZE[0]), Image.LANCZOS
    )
    original_arr = np.array(original_resized, dtype=np.float32)
    binary = (prob_mask >= PREDICTION_THRESHOLD).astype(np.float32)

    overlay = np.zeros_like(original_arr)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    mask_3d = np.stack([binary] * 3, axis=-1)
    result = original_arr * (1 - mask_3d * opacity) + overlay * mask_3d * opacity
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Draw green contour boundary
    if binary.any():
        dilated = ndimage.binary_dilation(binary > 0.5, iterations=1)
        eroded = ndimage.binary_erosion(binary > 0.5, iterations=1)
        boundary = dilated.astype(np.float32) - eroded.astype(np.float32) > 0
        result[boundary, 0] = 0
        result[boundary, 1] = 255
        result[boundary, 2] = 0

    return Image.fromarray(result)


def create_mask_image(prob_mask: np.ndarray) -> Image.Image:
    """Convert probability mask to a greyscale PIL image."""
    mask_uint8 = np.clip(prob_mask * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L")


# ---------------------------------------------------------------------------
# Main segmentation pipeline
# ---------------------------------------------------------------------------

class SegmentationService:
    """
    Brain Tumor Segmentation Service.

    Produces a segmentation mask and HONEST inference statistics.
    Never fabricates Dice, IoU, or accuracy values.
    """

    def __init__(self):
        self.input_size = INPUT_SIZE

    async def segment_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Main segmentation pipeline.

        Parameters
        ----------
        image_data : bytes  Raw image bytes (JPEG / PNG / etc.)

        Returns
        -------
        dict with keys:
            success                     bool
            overlay_image               str   (data-URI PNG)
            mask_image                  str   (data-URI PNG)
            inference_stats             dict  (tumor_detected, coverage, etc.)
            model_used                  str   'Enhanced U-Net' or 'SciPy Fallback'
            error                       str   (only on failure)
        """
        try:
            # --- Load image ---
            image = Image.open(io.BytesIO(image_data))

            # --- Try real model first ---
            model = _try_load_keras_model()
            if model is not None:
                img_batch = preprocess_image_for_inference(image)
                raw_pred = model.predict(img_batch, verbose=0)  # (1, H, W, 1)
                prob_mask = raw_pred[0, :, :, 0]               # (H, W)
                model_used = "Enhanced U-Net"
            else:
                # Fallback: SciPy classical segmentation
                img_rgb = np.array(
                    image.convert("RGB").resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
                prob_mask = _scipy_segment(img_rgb)
                model_used = "SciPy Fallback (no trained model loaded)"

            # --- Compute honest stats ---
            inference_stats = compute_inference_stats(prob_mask)

            # --- Build visual outputs ---
            overlay_image = create_overlay_image(image, prob_mask)
            mask_image = create_mask_image(prob_mask)

            # --- Encode to base64 data URIs ---
            import base64

            overlay_buf = io.BytesIO()
            overlay_image.save(overlay_buf, format="PNG")
            overlay_b64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

            mask_buf = io.BytesIO()
            mask_image.save(mask_buf, format="PNG")
            mask_b64 = base64.b64encode(mask_buf.getvalue()).decode("utf-8")

            return {
                "success": True,
                "overlay_image": f"data:image/png;base64,{overlay_b64}",
                "mask_image": f"data:image/png;base64,{mask_b64}",
                "inference_stats": inference_stats,
                "model_used": model_used,
            }

        except Exception as exc:
            logger.error("Segmentation error: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}