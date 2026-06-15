"""Utilities for visualizing semantic segmentation masks."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TIATOOLBOX_BCSS_OUTPUT_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (220, 20, 60),
    1: (70, 130, 180),
    2: (60, 179, 113),
    3: (214, 140, 40),
    4: (145, 82, 180),
}
CLASS_COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    **TIATOOLBOX_BCSS_OUTPUT_CLASS_COLORS,
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
    transparent_label_ids: set[int] | None = None,
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
    if transparent_label_ids is None:
        foreground = np.ones(label_mask.shape, dtype=bool)
    else:
        foreground = ~np.isin(label_mask, list(transparent_label_ids))
    output[foreground] = (1 - alpha) * output[foreground] + alpha * color_mask[foreground]
    return np.clip(output, 0, 255).astype(np.uint8)


def render_class_legend_image(
    legend: dict,
    min_width: int = 560,
    row_height: int = 34,
) -> Image.Image:
    """Render a compact visual legend with class ids, names, and pixel ratios."""
    classes = list(legend.get("classes", []))
    font = ImageFont.load_default()
    padding = 16
    swatch_size = 20
    header_height = 34
    width = max(min_width, 420)
    height = padding * 2 + header_height + max(1, len(classes)) * row_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title = "TIAToolbox BCSS grouped output legend"
    draw.text((padding, padding), title, fill=(20, 20, 20), font=font)
    y = padding + header_height

    if not classes:
        draw.text((padding, y), "No classes found.", fill=(80, 80, 80), font=font)
        return image

    for item in classes:
        color = tuple(int(value) for value in item["color_rgb"])
        class_id = item["class_id"]
        class_name = item["class_name"]
        pixel_ratio = float(item.get("pixel_ratio", 0.0))
        label = f"{class_id:<3} {class_name:<14} {pixel_ratio * 100:6.2f}%"

        draw.rectangle(
            [padding, y + 6, padding + swatch_size, y + 6 + swatch_size],
            fill=color,
            outline=(40, 40, 40),
        )
        draw.text((padding + swatch_size + 12, y + 8), label, fill=(20, 20, 20), font=font)
        y += row_height

    return image


def append_legend_to_image(
    rgb_image: Image.Image | np.ndarray,
    legend_image: Image.Image,
) -> Image.Image:
    """Append a legend image below an RGB image."""
    base = rgb_image.convert("RGB") if isinstance(rgb_image, Image.Image) else Image.fromarray(rgb_image)
    legend = legend_image.convert("RGB")
    width = max(base.width, legend.width)
    height = base.height + legend.height
    output = Image.new("RGB", (width, height), "white")
    output.paste(base, (0, 0))
    output.paste(legend, (0, base.height))
    return output
