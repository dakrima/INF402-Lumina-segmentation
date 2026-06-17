"""Patch extraction utilities for small in-memory images."""

from collections.abc import Iterator
from dataclasses import dataclass

from PIL import Image

EDGE_POLICIES = {"drop", "overlap", "pad"}


@dataclass(frozen=True)
class ExtractedPatch:
    """Patch image plus edge-handling metadata."""

    x: int
    y: int
    image: Image.Image
    original_width: int
    original_height: int
    padded: bool


def _validate_patch_parameters(patch_size: int, stride: int, edge_policy: str) -> None:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if edge_policy not in EDGE_POLICIES:
        allowed = ", ".join(sorted(EDGE_POLICIES))
        raise ValueError(f"edge_policy must be one of: {allowed}")


def _axis_positions(
    length: int,
    patch_size: int,
    stride: int,
    edge_policy: str,
) -> list[int]:
    """Return patch origins for one image axis under the requested edge policy."""
    if length <= 0:
        return []

    if edge_policy == "drop":
        if length < patch_size:
            return []
        return list(range(0, length - patch_size + 1, stride))

    if edge_policy == "overlap":
        if length < patch_size:
            return []
        step = min(stride, patch_size)
        positions = list(range(0, length - patch_size + 1, step))
        last_position = length - patch_size
        if not positions or positions[-1] != last_position:
            positions.append(last_position)
        return positions

    step = min(stride, patch_size)
    return list(range(0, length, step))


def generate_patch_coordinates(
    image_width: int,
    image_height: int,
    patch_size: int,
    stride: int,
    edge_policy: str,
) -> list[tuple[int, int]]:
    """Generate patch coordinates for a small image.

    Edge policies:
    - drop: keep only natural full patches and ignore incomplete borders.
    - overlap: shift the last full patch backward to cover the final border
      without padding. Images smaller than the patch size produce no patches.
    - pad: generate edge patches and pad them to patch_size with background.

    For cover-oriented policies, a stride larger than patch_size is capped to
    patch_size to avoid uncovered gaps in small-image experiments.
    """
    _validate_patch_parameters(patch_size, stride, edge_policy)
    if image_width <= 0 or image_height <= 0:
        return []

    x_positions = _axis_positions(image_width, patch_size, stride, edge_policy)
    y_positions = _axis_positions(image_height, patch_size, stride, edge_policy)
    return [(x, y) for y in y_positions for x in x_positions]


def iter_patches_with_metadata(
    image: Image.Image,
    patch_size: int,
    stride: int,
    edge_policy: str = "drop",
    pad_color: tuple[int, int, int] = (255, 255, 255),
) -> Iterator[ExtractedPatch]:
    """Yield patches with edge metadata for small in-memory images.

    Real WSI extraction will be integrated later with OpenSlide/TIAToolbox.
    """
    image = image.convert("RGB")
    width, height = image.size
    for x, y in generate_patch_coordinates(
        image_width=width,
        image_height=height,
        patch_size=patch_size,
        stride=stride,
        edge_policy=edge_policy,
    ):
        x_end = min(x + patch_size, width)
        y_end = min(y + patch_size, height)
        original_width = max(0, x_end - x)
        original_height = max(0, y_end - y)
        patch = image.crop((x, y, x_end, y_end))
        padded = original_width != patch_size or original_height != patch_size

        if padded:
            if edge_policy != "pad":
                raise ValueError("Only edge_policy='pad' can produce padded patches")
            padded_patch = Image.new("RGB", (patch_size, patch_size), pad_color)
            padded_patch.paste(patch, (0, 0))
            patch = padded_patch

        yield ExtractedPatch(
            x=x,
            y=y,
            image=patch,
            original_width=original_width,
            original_height=original_height,
            padded=padded,
        )


def iter_patches(
    image: Image.Image,
    patch_size: int,
    stride: int,
) -> Iterator[tuple[int, int, Image.Image]]:
    """Yield full-size patches from a small image with their top-left coordinates.

    This function is intended for early experiments on ordinary images. WSI
    extraction will be integrated later with OpenSlide/TIAToolbox.
    """
    for extracted_patch in iter_patches_with_metadata(
        image=image,
        patch_size=patch_size,
        stride=stride,
        edge_policy="drop",
    ):
        yield extracted_patch.x, extracted_patch.y, extracted_patch.image
