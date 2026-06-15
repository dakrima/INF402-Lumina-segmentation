"""Patch filtering utilities."""

import numpy as np
from PIL import Image

from src.preprocessing.tissue_detection import estimate_tissue_mask


def compute_tissue_ratio(
    rgb_patch: Image.Image | np.ndarray,
    background_threshold: int = 220,
) -> float:
    """Compute the approximate fraction of tissue in an RGB patch."""
    tissue_mask = estimate_tissue_mask(
        rgb_patch,
        background_threshold=background_threshold,
    )
    if tissue_mask.size == 0:
        return 0.0
    return float(np.mean(tissue_mask))
