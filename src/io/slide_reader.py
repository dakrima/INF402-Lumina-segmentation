"""Lightweight image reading utilities.

Real WSI reading will be integrated later with OpenSlide and TIAToolbox.
This module intentionally supports only common image files through PIL for
early local tests.
"""

from pathlib import Path

from PIL import Image


def read_image(path: str | Path) -> Image.Image:
    """Read a common image file and return it as an RGB PIL image.

    Parameters
    ----------
    path:
        Path to a small image file supported by PIL.

    Returns
    -------
    PIL.Image.Image
        RGB image loaded into memory.
    """
    image_path = Path(path)
    with Image.open(image_path) as image:
        return image.convert("RGB")
