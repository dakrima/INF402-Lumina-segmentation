"""Utilities for visualizing semantic segmentation masks."""

from __future__ import annotations

import numpy as np
from PIL import Image


CLASS_COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (70, 130, 180),
    2: (46, 160, 67),
    3: (214, 140, 40),
    4: (145, 82, 180),
    5: (210, 70, 70),
    6: (95, 180, 180),
    7: (180, 180, 80),
}
DEFAULT_LABEL_COLORS = np.array(
    [CLASS_COLOR_PALETTE[key] for key in sorted(CLASS_COLOR_PALETTE)],
    dtype=np.uint8,
)


def color_for_class_id(class_id: int) -> tuple[int, int, int]:
    """Return the RGB color used by masks and overlays for a class id."""
    palette_size = len(DEFAULT_LABEL_COLORS)
    color = DEFAULT_LABEL_COLORS[int(class_id) % palette_size]
    return int(color[0]), int(color[1]), int(color[2])


def normalize_label_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a model output mask to a 2D integer label mask."""
    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[-1] <= 32:
            array = np.argmax(array, axis=-1)
        elif array.shape[0] <= 32:
            array = np.argmax(array, axis=0)
        else:
            raise ValueError(f"Cannot infer class axis for prediction shape {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D label mask after normalization, got {array.shape}")
    return array.astype(np.int32, copy=False)


def resize_label_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D label mask to (width, height) using nearest-neighbor sampling."""
    label_mask = normalize_label_mask(mask)
    image = Image.fromarray(label_mask.astype(np.uint8), mode="L")
    resized = image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(resized).astype(np.int32, copy=False)


def colorize_label_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer labels to a reusable RGB palette for quick visual inspection."""
    label_mask = normalize_label_mask(mask)
    return DEFAULT_LABEL_COLORS[np.mod(label_mask, len(DEFAULT_LABEL_COLORS))]


def overlay_label_mask(
    rgb_image: Image.Image | np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a colorized label mask over an RGB image for technical review."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image = np.asarray(rgb_image.convert("RGB") if isinstance(rgb_image, Image.Image) else rgb_image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")

    label_mask = normalize_label_mask(mask)
    if label_mask.shape != image.shape[:2]:
        raise ValueError("mask must match the image height and width")

    color_mask = colorize_label_mask(label_mask).astype(np.float32)
    output = image[..., :3].astype(np.float32).copy()
    foreground = label_mask != 0
    output[foreground] = (1 - alpha) * output[foreground] + alpha * color_mask[foreground]
    return np.clip(output, 0, 255).astype(np.uint8)
