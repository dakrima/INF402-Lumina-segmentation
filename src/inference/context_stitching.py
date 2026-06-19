"""Geometry helpers for technical context-stitch inference probes."""

from __future__ import annotations

from typing import Any

import numpy as np


CLINICAL_WARNING = (
    "Technical segmentation/inference only. Not for diagnosis, not RCB, not clinical validation."
)
AXIS_CONVENTION = "WSI/PIL/OpenSlide uses (x, y); NumPy arrays use [y, x]"
PADDING_MODE_WHITE = "white"
WINDOW_IDS = ("window_00", "window_01", "window_10", "window_11")


def compute_padding(
    *,
    context_x0_requested: int,
    context_y0_requested: int,
    context_size: int,
    slide_width: int | None = None,
    slide_height: int | None = None,
    padding_mode: str = PADDING_MODE_WHITE,
) -> dict[str, Any]:
    """Compute white padding needed when a requested context falls outside a WSI."""
    if slide_width is None or slide_height is None:
        padding_left = padding_top = padding_right = padding_bottom = 0
    else:
        padding_left = max(0, -context_x0_requested)
        padding_top = max(0, -context_y0_requested)
        padding_right = max(0, context_x0_requested + context_size - slide_width)
        padding_bottom = max(0, context_y0_requested + context_size - slide_height)

    return {
        "context_padding_used": any(
            value > 0 for value in (padding_left, padding_right, padding_top, padding_bottom)
        ),
        "padding_left": int(padding_left),
        "padding_right": int(padding_right),
        "padding_top": int(padding_top),
        "padding_bottom": int(padding_bottom),
        "padding_mode": padding_mode,
    }


def compute_context_geometry(
    *,
    x_level0: int,
    y_level0: int,
    patch_size: int = 1024,
    patch_input_shape: int = 1024,
    patch_output_shape: int = 512,
    slide_width: int | None = None,
    slide_height: int | None = None,
) -> dict[str, Any]:
    """Return the 2x2 context-stitch geometry for a selected level-0 patch."""
    if patch_input_shape <= patch_output_shape:
        raise ValueError("patch_input_shape must be larger than patch_output_shape.")
    if (patch_input_shape - patch_output_shape) % 2 != 0:
        raise ValueError("patch_input_shape - patch_output_shape must be even.")
    if patch_size != patch_output_shape * 2:
        raise ValueError("context-stitch-2x2 currently expects patch_size == 2 * patch_output_shape.")

    margin = (patch_input_shape - patch_output_shape) // 2
    context_size = patch_size + 2 * margin
    target_x0 = int(x_level0)
    target_y0 = int(y_level0)
    target_x1 = target_x0 + patch_size
    target_y1 = target_y0 + patch_size
    context_x0_requested = target_x0 - margin
    context_y0_requested = target_y0 - margin

    padding = compute_padding(
        context_x0_requested=context_x0_requested,
        context_y0_requested=context_y0_requested,
        context_size=context_size,
        slide_width=slide_width,
        slide_height=slide_height,
    )
    read_x0 = max(0, context_x0_requested)
    read_y0 = max(0, context_y0_requested)
    read_x1 = (
        context_x0_requested + context_size
        if slide_width is None
        else min(slide_width, context_x0_requested + context_size)
    )
    read_y1 = (
        context_y0_requested + context_size
        if slide_height is None
        else min(slide_height, context_y0_requested + context_size)
    )

    window_offsets = {
        "window_00": (0, 0),
        "window_01": (patch_output_shape, 0),
        "window_10": (0, patch_output_shape),
        "window_11": (patch_output_shape, patch_output_shape),
    }
    windows: list[dict[str, Any]] = []
    for window_id in WINDOW_IDS:
        offset_x, offset_y = window_offsets[window_id]
        windows.append(
            {
                "window_id": window_id,
                "context_input_x0": offset_x,
                "context_input_y0": offset_y,
                "context_input_x1": offset_x + patch_input_shape,
                "context_input_y1": offset_y + patch_input_shape,
                "target_output_x0": offset_x,
                "target_output_y0": offset_y,
                "target_output_x1": offset_x + patch_output_shape,
                "target_output_y1": offset_y + patch_output_shape,
                "input_shape": [patch_input_shape, patch_input_shape],
                "output_shape": [patch_output_shape, patch_output_shape],
            }
        )

    return {
        "patch_input_shape": patch_input_shape,
        "patch_output_shape": patch_output_shape,
        "margin": margin,
        "target_patch_size": patch_size,
        "context_size": context_size,
        "target_x0": target_x0,
        "target_y0": target_y0,
        "target_x1": target_x1,
        "target_y1": target_y1,
        "context_x0_requested": context_x0_requested,
        "context_y0_requested": context_y0_requested,
        "context_width": context_size,
        "context_height": context_size,
        "read_x0": int(read_x0),
        "read_y0": int(read_y0),
        "read_x1": int(read_x1),
        "read_y1": int(read_y1),
        "read_width": int(max(0, read_x1 - read_x0)),
        "read_height": int(max(0, read_y1 - read_y0)),
        "padding": padding,
        "axis_convention": AXIS_CONVENTION,
        "windows": windows,
    }


def extract_windows_2x2(context_rgb: np.ndarray, *, patch_input_shape: int = 1024) -> dict[str, np.ndarray]:
    """Extract four 1024x1024 RGB windows from a 1536x1536 context image."""
    array = np.asarray(context_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("context_rgb must have shape [height, width, 3].")
    expected_context = patch_input_shape + patch_input_shape // 2
    if array.shape[:2] != (expected_context, expected_context):
        raise ValueError(
            "context_rgb has unexpected shape. "
            f"Expected {(expected_context, expected_context, 3)}, got {array.shape}."
        )

    step = patch_input_shape // 2
    return {
        "window_00": array[0:patch_input_shape, 0:patch_input_shape, :].copy(),
        "window_01": array[0:patch_input_shape, step:step + patch_input_shape, :].copy(),
        "window_10": array[step:step + patch_input_shape, 0:patch_input_shape, :].copy(),
        "window_11": array[step:step + patch_input_shape, step:step + patch_input_shape, :].copy(),
    }


def stitch_quadrants(outputs: dict[str, np.ndarray], *, patch_output_shape: int = 512) -> np.ndarray:
    """Stitch four 512x512 output label masks into a 1024x1024 target mask."""
    missing = [window_id for window_id in WINDOW_IDS if window_id not in outputs]
    if missing:
        raise ValueError(f"Missing output quadrants: {missing}")

    stitched = np.zeros((patch_output_shape * 2, patch_output_shape * 2), dtype=np.int32)
    placements = {
        "window_00": (0, 0),
        "window_01": (patch_output_shape, 0),
        "window_10": (0, patch_output_shape),
        "window_11": (patch_output_shape, patch_output_shape),
    }
    for window_id in WINDOW_IDS:
        output = np.asarray(outputs[window_id])
        output = np.squeeze(output)
        if output.shape != (patch_output_shape, patch_output_shape):
            raise ValueError(
                f"{window_id} must have shape {(patch_output_shape, patch_output_shape)}, "
                f"got {output.shape}."
            )
        target_x0, target_y0 = placements[window_id]
        stitched[
            target_y0:target_y0 + patch_output_shape,
            target_x0:target_x0 + patch_output_shape,
        ] = output.astype(np.int32, copy=False)
    return stitched
