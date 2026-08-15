"""
Shared metric functions for Enhanced U-Net training and inference.

These functions are used in:
  1. The Jupyter notebook for training (as Keras metrics / losses)
  2. The FastAPI backend for loading the .keras model (custom_objects)
  3. The /api/evaluate endpoint for computing real supervised metrics

ALL metric functions use smooth=1e-6 (not 100) to avoid inflating scores.
"""

import numpy as np

# ---------------------------------------------------------------------------
# TensorFlow / Keras metrics  (imported lazily so the module can also be used
# in pure-NumPy contexts, e.g. unit tests)
# ---------------------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Soft Dice coefficient (per-sample, averaged over the batch).

    Dice = (2 * |intersection| + ε) / (|y_true| + |y_pred| + ε)

    Works with soft (sigmoid) predictions during training and with
    thresholded binary predictions during evaluation.
    """
    import tensorflow.keras.backend as K

    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    return K.mean(
        (2.0 * intersection + smooth)
        / (K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1) + smooth)
    )


def dice_loss(y_true, y_pred):
    """Dice loss: approaches 0 for perfect segmentation."""
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy + Dice Loss.

    total_loss = BCE + (1 - Dice)
    """
    import tensorflow as tf

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1e-6):
    """
    Intersection-over-Union (Jaccard index), per-sample averaged.

    IoU = (intersection + ε) / (union + ε)
    """
    import tensorflow.keras.backend as K

    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1) - intersection
    return K.mean((intersection + smooth) / (union + smooth))


# ---------------------------------------------------------------------------
# Pure-NumPy metric helpers  (for backend evaluation without TF dependency)
# ---------------------------------------------------------------------------

def binary_dice_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Binary Dice on flat NumPy arrays. Both must be {0, 1}."""
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    intersection = np.sum(y_true * y_pred)
    return float((2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth))


def binary_iou_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Binary IoU on flat NumPy arrays. Both must be {0, 1}."""
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def pixel_precision_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Precision = TP / (TP + FP)."""
    tp = np.sum((y_true == 1) & (y_pred == 1)).astype(np.float64)
    fp = np.sum((y_true == 0) & (y_pred == 1)).astype(np.float64)
    return float((tp + smooth) / (tp + fp + smooth))


def pixel_recall_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Recall / Sensitivity = TP / (TP + FN)."""
    tp = np.sum((y_true == 1) & (y_pred == 1)).astype(np.float64)
    fn = np.sum((y_true == 1) & (y_pred == 0)).astype(np.float64)
    return float((tp + smooth) / (tp + fn + smooth))


def pixel_specificity_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Specificity = TN / (TN + FP)."""
    tn = np.sum((y_true == 0) & (y_pred == 0)).astype(np.float64)
    fp = np.sum((y_true == 0) & (y_pred == 1)).astype(np.float64)
    return float((tn + smooth) / (tn + fp + smooth))


def pixel_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pixel accuracy = (TP + TN) / total."""
    correct = np.sum(y_true == y_pred).astype(np.float64)
    total = float(y_true.size)
    return float(correct / total) if total > 0 else 0.0


def compute_all_metrics_np(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5):
    """
    Compute all supervised metrics from NumPy arrays.

    Parameters
    ----------
    y_true     : np.ndarray  binary ground-truth mask (H, W), values in {0, 1}
    y_pred_prob: np.ndarray  probability mask (H, W), values in [0, 1]
    threshold  : float       binarisation threshold

    Returns
    -------
    dict with dice, iou, precision, recall, specificity, pixel_accuracy
    """
    y_pred_bin = (y_pred_prob >= threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)
    return {
        "dice": round(binary_dice_np(y_true_bin, y_pred_bin), 6),
        "iou": round(binary_iou_np(y_true_bin, y_pred_bin), 6),
        "precision": round(pixel_precision_np(y_true_bin, y_pred_bin), 6),
        "recall": round(pixel_recall_np(y_true_bin, y_pred_bin), 6),
        "specificity": round(pixel_specificity_np(y_true_bin, y_pred_bin), 6),
        "pixel_accuracy": round(pixel_accuracy_np(y_true_bin, y_pred_bin), 6),
    }
