"""Simple tissue detection baselines."""

import numpy as np
from PIL import Image


def estimate_tissue_mask(
    rgb_image: Image.Image | np.ndarray,
    background_threshold: int = 220,
) -> np.ndarray:
    """Estimate a rough tissue mask by thresholding bright background.

    This is a simple computational baseline for early patch filtering. It is
    not a clinical tissue detector and must not be interpreted as a definitive
    decision about healthy or pathological tissue.

    Parameters
    ----------
    rgb_image:
        RGB image as a PIL image or NumPy array.
    background_threshold:
        Pixels with all channels above this threshold are treated as likely
        background.

    Returns
    -------
    numpy.ndarray
        Boolean mask where True indicates likely tissue.
    """
    array = np.asarray(rgb_image.convert("RGB") if isinstance(rgb_image, Image.Image) else rgb_image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")

    rgb = array[..., :3]
    return np.any(rgb < background_threshold, axis=-1)
