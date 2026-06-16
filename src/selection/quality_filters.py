"""CPU-friendly feature extraction for smart patch selection."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.preprocessing.wsi_patch_extraction import (
    compute_simple_tissue_ratio,
    estimate_thumbnail_tissue_mask,
)


NUCLEAR_SIGNAL_METHOD = "purple_hematoxylin_proxy_v1"
VISUAL_ENTROPY_METHOD = "grayscale_histogram_entropy_32_bins"
BLUR_SCORE_METHOD = "grayscale_gradient_variance"
ARTIFACT_PENALTY_METHOD = "white_black_saturation_low_entropy_heuristic"


def _resize_for_features(rgb_image: Image.Image, feature_size: int) -> Image.Image:
    if feature_size <= 0:
        raise ValueError("feature_size must be positive.")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return rgb_image.convert("RGB").resize((feature_size, feature_size), resampling)


def _rgb_array_01(rgb_image: Image.Image) -> np.ndarray:
    return np.asarray(rgb_image.convert("RGB"), dtype=np.float32) / 255.0


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return values.reshape(-1)
    return values[mask]


def compute_nuclear_signal(rgb_array: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Approximate hematoxylin/nuclear signal from purple-blue dark pixels."""
    red = rgb_array[..., 0]
    green = rgb_array[..., 1]
    blue = rgb_array[..., 2]
    brightness = np.mean(rgb_array, axis=-1)
    saturation = np.max(rgb_array, axis=-1) - np.min(rgb_array, axis=-1)
    purple_component = np.maximum(0.0, ((red + blue) * 0.5) - green)
    purple_score = purple_component * saturation * (1.0 - brightness)
    values = _masked_values(purple_score, tissue_mask)
    if values.size == 0:
        return 0.0
    return float(np.clip(np.mean(values) * 4.0, 0.0, 1.0))


def compute_visual_entropy(
    rgb_array: np.ndarray,
    tissue_mask: np.ndarray,
    bins: int = 32,
) -> float:
    """Compute normalized grayscale entropy, preferably inside tissue."""
    gray = (
        0.299 * rgb_array[..., 0]
        + 0.587 * rgb_array[..., 1]
        + 0.114 * rgb_array[..., 2]
    )
    values = _masked_values(gray, tissue_mask)
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(np.sum(hist))
    if total == 0:
        return 0.0
    probabilities = hist[hist > 0].astype(np.float64) / total
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    max_entropy = float(np.log2(bins))
    if max_entropy <= 0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_blur_score(rgb_array: np.ndarray) -> float:
    """Estimate sharpness with grayscale gradient variance."""
    gray = (
        0.299 * rgb_array[..., 0]
        + 0.587 * rgb_array[..., 1]
        + 0.114 * rgb_array[..., 2]
    )
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    return float(max(0.0, np.var(dx) + np.var(dy)))


def compute_artifact_penalty(
    rgb_array: np.ndarray,
    tissue_ratio: float,
    visual_entropy: float,
) -> float:
    """Penalize simple non-informative or artifact-like visual patterns."""
    brightness = np.mean(rgb_array, axis=-1)
    saturation = np.max(rgb_array, axis=-1) - np.min(rgb_array, axis=-1)

    white_fraction = float(np.mean((brightness > 0.92) & (saturation < 0.08)))
    black_fraction = float(np.mean(brightness < 0.05))
    extreme_saturation_fraction = float(np.mean(saturation > 0.80))
    low_tissue_penalty = float(np.clip((0.20 - tissue_ratio) / 0.20, 0.0, 1.0))
    low_entropy_penalty = float(np.clip((0.15 - visual_entropy) / 0.15, 0.0, 1.0))

    penalty = (
        0.35 * white_fraction
        + 0.20 * black_fraction
        + 0.15 * extreme_saturation_fraction
        + 0.15 * low_tissue_penalty
        + 0.15 * low_entropy_penalty
    )
    return float(np.clip(penalty, 0.0, 1.0))


def compute_patch_features(
    rgb_patch: Image.Image,
    feature_size: int = 256,
) -> dict[str, float]:
    """Compute all smart-selector features on a downsampled patch."""
    feature_patch = _resize_for_features(rgb_patch, feature_size=feature_size)
    rgb_array = _rgb_array_01(feature_patch)
    tissue_mask = estimate_thumbnail_tissue_mask(feature_patch)
    tissue_ratio = compute_simple_tissue_ratio(feature_patch)
    nuclear_signal = compute_nuclear_signal(
        rgb_array=rgb_array,
        tissue_mask=tissue_mask,
    )
    visual_entropy = compute_visual_entropy(
        rgb_array=rgb_array,
        tissue_mask=tissue_mask,
    )
    blur_score = compute_blur_score(rgb_array)
    artifact_penalty = compute_artifact_penalty(
        rgb_array=rgb_array,
        tissue_ratio=tissue_ratio,
        visual_entropy=visual_entropy,
    )
    return {
        "tissue_ratio": tissue_ratio,
        "nuclear_signal": nuclear_signal,
        "visual_entropy": visual_entropy,
        "blur_score": blur_score,
        "artifact_penalty": artifact_penalty,
    }
