"""NumPy metrics for semantic segmentation."""

import numpy as np


def _valid_mask(y_true: np.ndarray, ignore_index: int | None) -> np.ndarray:
    if ignore_index is None:
        return np.ones_like(y_true, dtype=bool)
    return y_true != ignore_index


def pixel_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_index: int | None = None,
) -> float:
    """Compute pixel accuracy, optionally ignoring a label value."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    valid = _valid_mask(true, ignore_index)
    if not np.any(valid):
        return float("nan")

    return float(np.mean(true[valid] == pred[valid]))


def mean_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Compute mean intersection-over-union across classes.

    Classes with zero union are skipped. If no class has a valid union, the
    function returns NaN.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    valid = _valid_mask(true, ignore_index)
    ious: list[float] = []
    for class_id in range(num_classes):
        true_class = (true == class_id) & valid
        pred_class = (pred == class_id) & valid
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        if union > 0:
            ious.append(float(intersection / union))

    if not ious:
        return float("nan")
    return float(np.mean(ious))
