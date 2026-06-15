"""Patch extraction utilities for small in-memory images."""

from collections.abc import Iterator

from PIL import Image


def iter_patches(
    image: Image.Image,
    patch_size: int,
    stride: int,
) -> Iterator[tuple[int, int, Image.Image]]:
    """Yield full-size patches from a small image with their top-left coordinates.

    This function is intended for early experiments on ordinary images. WSI
    extraction will be integrated later with OpenSlide/TIAToolbox.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    width, height = image.size
    if width < patch_size or height < patch_size:
        return

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield x, y, image.crop((x, y, x + patch_size, y + patch_size))
