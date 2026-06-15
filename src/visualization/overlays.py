"""Overlay helpers for quick segmentation visualization."""

import numpy as np
from PIL import Image


def create_overlay(
    rgb_image: Image.Image | np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a simple red overlay for nonzero mask pixels.

    Parameters
    ----------
    rgb_image:
        RGB image as PIL image or NumPy array.
    mask:
        Boolean or integer mask. Nonzero values are highlighted.
    alpha:
        Overlay opacity in the range [0, 1].

    Returns
    -------
    numpy.ndarray
        RGB uint8 image with the overlay applied.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image = np.asarray(rgb_image.convert("RGB") if isinstance(rgb_image, Image.Image) else rgb_image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")

    binary_mask = np.asarray(mask) != 0
    if binary_mask.shape != image.shape[:2]:
        raise ValueError("mask must match the image height and width")

    output = image[..., :3].astype(np.float32).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    output[binary_mask] = (1 - alpha) * output[binary_mask] + alpha * red
    return np.clip(output, 0, 255).astype(np.uint8)
