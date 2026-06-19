"""Input validation helpers for technical patch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPECTED_MODEL_MPP = 0.25
MPP_TOLERANCE = 0.05
EXPECTED_INPUT_WIDTH = 1024
EXPECTED_INPUT_HEIGHT = 1024

MPP_UNAVAILABLE_WARNING = (
    "MPP metadata unavailable; scale compatibility with model could not be confirmed."
)
MPP_MISMATCH_WARNING = (
    "Input MPP differs from expected model MPP; segmentation output may be less reliable technically."
)


def _safe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _optional_abs_diff(value: float | None, expected: float) -> float | None:
    if value is None:
        return None
    return float(abs(value - expected))


def _json_number(value: np.generic | int | float) -> int | float:
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_mpp_validation(
    metadata: dict[str, object] | None,
    *,
    expected_model_mpp: float = EXPECTED_MODEL_MPP,
    mpp_tolerance: float = MPP_TOLERANCE,
) -> dict[str, object]:
    """Build scale metadata validation without making clinical claims."""
    metadata = metadata or {}
    input_mpp_x = _safe_float(metadata.get("mpp_x") or metadata.get("input_mpp_x"))
    input_mpp_y = _safe_float(metadata.get("mpp_y") or metadata.get("input_mpp_y"))
    mpp_x_abs_diff = _optional_abs_diff(input_mpp_x, expected_model_mpp)
    mpp_y_abs_diff = _optional_abs_diff(input_mpp_y, expected_model_mpp)
    available_diffs = [
        diff for diff in (mpp_x_abs_diff, mpp_y_abs_diff)
        if diff is not None
    ]
    mpp_available = bool(available_diffs)
    mpp_within_tolerance = (
        bool(available_diffs)
        and all(diff <= mpp_tolerance for diff in available_diffs)
    )

    warnings: list[str] = []
    if not mpp_available:
        warnings.append(MPP_UNAVAILABLE_WARNING)
    elif not mpp_within_tolerance:
        warnings.append(MPP_MISMATCH_WARNING)

    return {
        "expected_model_mpp": expected_model_mpp,
        "input_mpp_x": input_mpp_x,
        "input_mpp_y": input_mpp_y,
        "mpp_x_abs_diff": mpp_x_abs_diff,
        "mpp_y_abs_diff": mpp_y_abs_diff,
        "mpp_tolerance": mpp_tolerance,
        "mpp_available": mpp_available,
        "mpp_within_tolerance": mpp_within_tolerance,
        "warnings": warnings,
    }


def _first_non_empty(*values: object) -> object:
    for value in values:
        if value not in ("", None):
            return value
    return ""


def build_selection_metadata_for_patch(
    *,
    selected_row: dict[str, str],
    selection_summary: dict[str, Any],
    input_selection_dir: Path,
    selected_metadata_path: Path,
    selection_summary_path: Path,
) -> dict[str, object]:
    """Flatten selector metadata needed to trace a segmented patch."""
    metadata: dict[str, object] = {
        "selector": _first_non_empty(
            selection_summary.get("selector"),
            selected_row.get("selector"),
            selected_row.get("selection_method"),
        ),
        "selection_method": _first_non_empty(
            selected_row.get("selection_method"),
            selection_summary.get("selection_method"),
            selection_summary.get("selector"),
        ),
        "wsi_path": _first_non_empty(
            selected_row.get("source_wsi_path"),
            selection_summary.get("wsi_path"),
        ),
        "source_wsi_path": _first_non_empty(
            selected_row.get("source_wsi_path"),
            selection_summary.get("wsi_path"),
        ),
        "x_level0": selected_row.get("x_level0", ""),
        "y_level0": selected_row.get("y_level0", ""),
        "patch_size": _first_non_empty(
            selected_row.get("patch_size"),
            selection_summary.get("patch_size"),
        ),
        "stride": _first_non_empty(
            selected_row.get("stride"),
            selection_summary.get("stride"),
        ),
        "tiatoolbox_index": selected_row.get("tiatoolbox_index", ""),
        "candidate_pool": selection_summary.get("candidate_pool", ""),
        "tissue_mask_method": _first_non_empty(
            selected_row.get("tissue_mask_method"),
            selection_summary.get("tissue_mask_method"),
        ),
        "mpp_x": _first_non_empty(
            selected_row.get("mpp_x"),
            selection_summary.get("mpp_x"),
        ),
        "mpp_y": _first_non_empty(
            selected_row.get("mpp_y"),
            selection_summary.get("mpp_y"),
        ),
        "objective_power": _first_non_empty(
            selected_row.get("objective_power"),
            selection_summary.get("objective_power"),
        ),
        "source_selection_dir": str(input_selection_dir),
        "source_selected_metadata_csv": str(selected_metadata_path),
        "source_selection_summary_json": str(selection_summary_path),
    }

    required_fields = [
        "selector",
        "selection_method",
        "wsi_path",
        "x_level0",
        "y_level0",
        "patch_size",
        "stride",
        "tiatoolbox_index",
        "candidate_pool",
        "tissue_mask_method",
        "mpp_x",
        "mpp_y",
        "objective_power",
    ]
    missing_fields = [
        field_name for field_name in required_fields
        if metadata.get(field_name) in ("", None)
    ]
    metadata["metadata_warnings"] = [
        f"Selection metadata field unavailable: {field_name}."
        for field_name in missing_fields
    ]
    return metadata


def validate_patch_input(
    image_path: Path,
    *,
    expected_width: int = EXPECTED_INPUT_WIDTH,
    expected_height: int = EXPECTED_INPUT_HEIGHT,
    selection_metadata: dict[str, object] | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    """Validate a patch image and return an RGB image plus JSON-safe metadata."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Patch image path does not exist: {image_path}")

    warnings: list[str] = []
    try:
        with Image.open(image_path) as image:
            original_mode = image.mode
            rgb_image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - preserve PIL diagnostic
        raise RuntimeError(f"Patch image could not be opened with PIL: {image_path} ({exc})") from exc

    converted_to_rgb = original_mode != "RGB"
    if converted_to_rgb:
        warnings.append(f"Image mode {original_mode} was converted to RGB for inference.")

    input_width, input_height = rgb_image.size
    input_shape_matches_model = (
        input_width == expected_width
        and input_height == expected_height
    )
    if not input_shape_matches_model:
        warnings.append(
            "Input patch size differs from expected model input shape; "
            "the image was not resized before inference."
        )

    array = np.asarray(rgb_image)
    is_uint8 = array.dtype == np.uint8
    if not is_uint8:
        warnings.append(f"Input array dtype is {array.dtype}, expected uint8.")
    input_min = _json_number(np.min(array))
    input_max = _json_number(np.max(array))
    input_mean = float(np.mean(array))
    input_std = float(np.std(array))
    range_looks_valid = 0 <= float(input_min) <= 255 and 0 <= float(input_max) <= 255
    if not range_looks_valid:
        warnings.append("Input array range is outside expected 0-255 bounds.")
    num_channels = int(array.shape[2]) if array.ndim == 3 else 1
    if num_channels != 3:
        warnings.append(f"Input array has {num_channels} channels after RGB conversion.")

    mpp_validation = build_mpp_validation(selection_metadata)
    warnings.extend(str(warning) for warning in mpp_validation["warnings"])

    strict_validation_failed = (
        not input_shape_matches_model
        or not is_uint8
        or not range_looks_valid
        or num_channels != 3
    )

    validation = {
        "status": "completed_with_warnings" if warnings else "completed",
        "image_path": str(image_path),
        "file_exists": True,
        "pil_opened": True,
        "original_mode": original_mode,
        "converted_to_rgb": converted_to_rgb,
        "input_width": input_width,
        "input_height": input_height,
        "expected_input_width": expected_width,
        "expected_input_height": expected_height,
        "input_shape_matches_model": input_shape_matches_model,
        "input_dtype": str(array.dtype),
        "input_min": input_min,
        "input_max": input_max,
        "input_mean": input_mean,
        "input_std": input_std,
        "num_channels": num_channels,
        "is_uint8": is_uint8,
        "range_looks_valid": range_looks_valid,
        "validation_array_source": "post_rgb_conversion",
        "mpp_validation": mpp_validation,
        "strict_validation_failed": strict_validation_failed,
        "warnings": warnings,
    }
    return rgb_image, validation
