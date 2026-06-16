"""Candidate generation utilities for WSI patch selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.preprocessing.wsi_patch_extraction import estimate_thumbnail_tissue_mask


@dataclass(frozen=True)
class PatchCandidate:
    """Level-0 candidate coordinate with thumbnail tissue estimate."""

    candidate_id: str
    grid_index: int
    x_level0: int
    y_level0: int
    patch_size: int
    thumbnail_tissue_ratio: float


def generate_grid_coordinates(
    slide_width: int,
    slide_height: int,
    patch_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Generate deterministic level-0 grid coordinates in row-major order."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if slide_width < patch_size or slide_height < patch_size:
        return []

    x_positions = range(0, slide_width - patch_size + 1, stride)
    y_positions = range(0, slide_height - patch_size + 1, stride)
    return [(x_level0, y_level0) for y_level0 in y_positions for x_level0 in x_positions]


def thumbnail_bbox_for_patch(
    x_level0: int,
    y_level0: int,
    patch_size: int,
    slide_dimensions: tuple[int, int],
    thumbnail_dimensions: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a level-0 patch rectangle to thumbnail pixel coordinates."""
    slide_width, slide_height = slide_dimensions
    thumb_width, thumb_height = thumbnail_dimensions
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height

    x0 = int(np.floor(x_level0 * scale_x))
    y0 = int(np.floor(y_level0 * scale_y))
    x1 = int(np.ceil((x_level0 + patch_size) * scale_x))
    y1 = int(np.ceil((y_level0 + patch_size) * scale_y))

    x0 = max(0, min(thumb_width - 1, x0))
    y0 = max(0, min(thumb_height - 1, y0))
    x1 = max(x0 + 1, min(thumb_width, x1))
    y1 = max(y0 + 1, min(thumb_height, y1))
    return x0, y0, x1, y1


def compute_thumbnail_tissue_ratio(
    tissue_mask: np.ndarray,
    x_level0: int,
    y_level0: int,
    patch_size: int,
    slide_dimensions: tuple[int, int],
) -> float:
    """Compute the approximate tissue ratio for a candidate on a thumbnail mask."""
    thumbnail_dimensions = (tissue_mask.shape[1], tissue_mask.shape[0])
    x0, y0, x1, y1 = thumbnail_bbox_for_patch(
        x_level0=x_level0,
        y_level0=y_level0,
        patch_size=patch_size,
        slide_dimensions=slide_dimensions,
        thumbnail_dimensions=thumbnail_dimensions,
    )
    thumbnail_patch_mask = tissue_mask[y0:y1, x0:x1]
    if thumbnail_patch_mask.size == 0:
        return 0.0
    return float(np.mean(thumbnail_patch_mask))


def generate_tissue_candidates(
    thumbnail: Image.Image,
    slide_dimensions: tuple[int, int],
    patch_size: int,
    stride: int,
    min_tissue_ratio: float,
) -> tuple[list[PatchCandidate], int]:
    """Generate grid candidates and keep those passing the thumbnail tissue mask."""
    slide_width, slide_height = slide_dimensions
    tissue_mask = estimate_thumbnail_tissue_mask(thumbnail)
    grid_coordinates = generate_grid_coordinates(
        slide_width=slide_width,
        slide_height=slide_height,
        patch_size=patch_size,
        stride=stride,
    )

    candidates: list[PatchCandidate] = []
    for grid_index, (x_level0, y_level0) in enumerate(grid_coordinates):
        thumbnail_tissue_ratio = compute_thumbnail_tissue_ratio(
            tissue_mask=tissue_mask,
            x_level0=x_level0,
            y_level0=y_level0,
            patch_size=patch_size,
            slide_dimensions=slide_dimensions,
        )
        if thumbnail_tissue_ratio < min_tissue_ratio:
            continue
        candidates.append(
            PatchCandidate(
                candidate_id=f"candidate_{grid_index:06d}",
                grid_index=grid_index,
                x_level0=x_level0,
                y_level0=y_level0,
                patch_size=patch_size,
                thumbnail_tissue_ratio=thumbnail_tissue_ratio,
            )
        )

    return candidates, len(grid_coordinates)
