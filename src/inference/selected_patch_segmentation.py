"""Run technical segmentation over patches selected by a selector run."""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.inference.input_validation import (
    EXPECTED_INPUT_HEIGHT,
    EXPECTED_INPUT_WIDTH,
    EXPECTED_MODEL_MPP,
    MPP_TOLERANCE,
    build_selection_metadata_for_patch,
)


CLINICAL_WARNING = (
    "Technical segmentation/inference only. Not for diagnosis, not RCB, not clinical validation."
)
PREDICTION_RESOLUTION_NOTE = (
    "Class counts and ratios are computed on the raw prediction mask. "
    "Visual masks and overlays may be resized with nearest-neighbor interpolation "
    "only for visual inspection over the input patch."
)
REQUIRED_SELECTION_FILES = [
    "selected_metadata.csv",
    "selection_summary.json",
    "method_config.json",
]
REQUIRED_SELECTION_DIRS = ["selected"]
PER_PATCH_SEGMENTATION_FIELDS = [
    "patch_id",
    "filename",
    "rank",
    "x_level0",
    "y_level0",
    "patch_size",
    "selection_method",
    "source_wsi_path",
    "status",
    "error",
    "resolved_device",
    "model_name",
    "prediction_shape",
    "visualized_mask_shape",
    "unique_prediction_values",
    "class_pixel_counts",
    "class_pixel_ratios",
    "class_count_source",
    "raw_prediction_total_pixels",
    "visualized_prediction_total_pixels",
    "probability_summary",
    "mean_max_probability",
    "median_max_probability",
    "min_max_probability",
    "max_max_probability",
    "mask_path",
    "prediction_mask_raw_path",
    "prediction_mask_visual_path",
    "prediction_labels_raw_npy_path",
    "prediction_labels_visual_npy_path",
    "prediction_probabilities_npz_path",
    "overlay_path",
    "overlay_with_legend_path",
    "input_preview_path",
    "patch_inference_summary_path",
    "input_image_shape",
    "raw_prediction_shape",
    "resized_for_visualization",
    "num_patch_warnings",
    "patch_warnings",
    "selector",
    "wsi_path",
    "stride",
    "tiatoolbox_index",
    "candidate_pool",
    "tissue_mask_method",
    "mpp_x",
    "mpp_y",
    "objective_power",
    "source_selection_dir",
    "source_selected_metadata_csv",
    "source_selection_summary_json",
    "selection_metadata_warnings",
    "input_validation_status",
    "file_exists",
    "pil_opened",
    "original_mode",
    "converted_to_rgb",
    "input_width",
    "input_height",
    "expected_input_width",
    "expected_input_height",
    "input_shape_matches_model",
    "input_dtype",
    "input_min",
    "input_max",
    "input_mean",
    "input_std",
    "num_channels",
    "is_uint8",
    "range_looks_valid",
    "expected_model_mpp",
    "input_mpp_x",
    "input_mpp_y",
    "mpp_x_abs_diff",
    "mpp_y_abs_diff",
    "mpp_tolerance",
    "mpp_available",
    "mpp_within_tolerance",
    "input_validation_warnings",
]


@dataclass(frozen=True)
class SelectedPatchSegmentationConfig:
    """Configuration for segmenting already selected patches."""

    input_selection_dir: Path
    output_dir: Path
    root_dir: Path
    model_name: str
    requested_device: str = "auto"
    input_mode: str = "patch"
    overlay_alpha: float = 0.45
    limit_patches: int | None = None
    overwrite: bool = False
    strict_input_validation: bool = False
    save_probabilities: bool = False
    save_visual_labels_npy: bool = False


def _resolve_path(path: Path, root_dir: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (root_dir / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _has_user_outputs(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(child.name != ".gitkeep" for child in output_dir.iterdir())


def _prepare_output_dir(output_dir: Path, root_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    if _has_user_outputs(resolved_output) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {resolved_output}. "
            "Use --overwrite to regenerate this segmentation run."
        )

    if overwrite and resolved_output.exists():
        dangerous_paths = {
            Path("/").resolve(),
            Path.home().resolve(),
            resolved_root,
            resolved_root / "data",
            resolved_root / "outputs",
        }
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite only clears output directories inside the repository.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
        shutil.rmtree(resolved_output)

    for child_dir in [
        output_dir,
        output_dir / "masks",
        output_dir / "masks_raw",
        output_dir / "masks_visual",
        output_dir / "labels_raw",
        output_dir / "labels_visual",
        output_dir / "overlays",
        output_dir / "overlays_with_legend",
        output_dir / "input_previews",
        output_dir / "per_patch",
    ]:
        child_dir.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_input_selection_dir(input_selection_dir: Path) -> None:
    if not input_selection_dir.exists() or not input_selection_dir.is_dir():
        raise FileNotFoundError(f"Input selection directory does not exist: {input_selection_dir}")
    for file_name in REQUIRED_SELECTION_FILES:
        path = input_selection_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing required selection file: {path}")
    for dir_name in REQUIRED_SELECTION_DIRS:
        path = input_selection_dir / dir_name
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Missing required selection directory: {path}")


def _json_cell(value: object) -> str:
    if value in ("", None):
        return ""
    return json.dumps(value, sort_keys=True)


def _bool_cell(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _shape_from_image_size(summary: dict[str, Any]) -> list[int] | None:
    image_height = summary.get("image_height")
    image_width = summary.get("image_width")
    if image_height is None or image_width is None:
        return None
    try:
        return [int(image_height), int(image_width)]
    except (TypeError, ValueError):
        return None


def _summary_was_resized_for_visualization(summary: dict[str, Any]) -> bool:
    prediction_shape = summary.get("prediction_shape")
    visualized_shape = summary.get("visualized_mask_shape")
    if prediction_shape and visualized_shape and prediction_shape != visualized_shape:
        return True
    warnings = summary.get("warnings") or []
    return any("resized with nearest neighbor" in str(warning).lower() for warning in warnings)


def _selection_metadata_cells(selection_metadata: dict[str, object] | None) -> dict[str, object]:
    selection_metadata = selection_metadata or {}
    return {
        "selector": selection_metadata.get("selector", ""),
        "wsi_path": selection_metadata.get("wsi_path", ""),
        "stride": selection_metadata.get("stride", ""),
        "tiatoolbox_index": selection_metadata.get("tiatoolbox_index", ""),
        "candidate_pool": selection_metadata.get("candidate_pool", ""),
        "tissue_mask_method": selection_metadata.get("tissue_mask_method", ""),
        "mpp_x": selection_metadata.get("mpp_x", ""),
        "mpp_y": selection_metadata.get("mpp_y", ""),
        "objective_power": selection_metadata.get("objective_power", ""),
        "source_selection_dir": selection_metadata.get("source_selection_dir", ""),
        "source_selected_metadata_csv": selection_metadata.get("source_selected_metadata_csv", ""),
        "source_selection_summary_json": selection_metadata.get("source_selection_summary_json", ""),
        "selection_metadata_warnings": _json_cell(
            selection_metadata.get("metadata_warnings", [])
        ),
    }


def _input_validation_cells(input_validation: dict[str, object] | None) -> dict[str, object]:
    input_validation = input_validation or {}
    mpp_validation = input_validation.get("mpp_validation")
    if not isinstance(mpp_validation, dict):
        mpp_validation = {}
    return {
        "input_validation_status": input_validation.get("status", ""),
        "file_exists": _bool_cell(input_validation.get("file_exists")),
        "pil_opened": _bool_cell(input_validation.get("pil_opened")),
        "original_mode": input_validation.get("original_mode", ""),
        "converted_to_rgb": _bool_cell(input_validation.get("converted_to_rgb")),
        "input_width": input_validation.get("input_width", ""),
        "input_height": input_validation.get("input_height", ""),
        "expected_input_width": input_validation.get("expected_input_width", ""),
        "expected_input_height": input_validation.get("expected_input_height", ""),
        "input_shape_matches_model": _bool_cell(
            input_validation.get("input_shape_matches_model")
        ),
        "input_dtype": input_validation.get("input_dtype", ""),
        "input_min": input_validation.get("input_min", ""),
        "input_max": input_validation.get("input_max", ""),
        "input_mean": input_validation.get("input_mean", ""),
        "input_std": input_validation.get("input_std", ""),
        "num_channels": input_validation.get("num_channels", ""),
        "is_uint8": _bool_cell(input_validation.get("is_uint8")),
        "range_looks_valid": _bool_cell(input_validation.get("range_looks_valid")),
        "expected_model_mpp": mpp_validation.get("expected_model_mpp", ""),
        "input_mpp_x": mpp_validation.get("input_mpp_x", ""),
        "input_mpp_y": mpp_validation.get("input_mpp_y", ""),
        "mpp_x_abs_diff": mpp_validation.get("mpp_x_abs_diff", ""),
        "mpp_y_abs_diff": mpp_validation.get("mpp_y_abs_diff", ""),
        "mpp_tolerance": mpp_validation.get("mpp_tolerance", ""),
        "mpp_available": _bool_cell(mpp_validation.get("mpp_available")),
        "mpp_within_tolerance": _bool_cell(mpp_validation.get("mpp_within_tolerance")),
        "input_validation_warnings": _json_cell(input_validation.get("warnings", [])),
    }


def _patch_id_for_row(row: dict[str, str]) -> str:
    patch_id = row.get("patch_id", "").strip()
    if patch_id:
        return patch_id
    filename = row.get("filename", "").strip()
    return Path(filename).stem


def _empty_result_row(
    source_row: dict[str, str],
    *,
    patch_id: str,
    status: str,
    error: str = "",
    selection_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "filename": source_row.get("filename", ""),
        "rank": source_row.get("rank", ""),
        "x_level0": source_row.get("x_level0", ""),
        "y_level0": source_row.get("y_level0", ""),
        "patch_size": source_row.get("patch_size", ""),
        "selection_method": source_row.get("selection_method", ""),
        "source_wsi_path": source_row.get("source_wsi_path", ""),
        "status": status,
        "error": error,
        "resolved_device": "",
        "model_name": "",
        "prediction_shape": "",
        "visualized_mask_shape": "",
        "unique_prediction_values": "",
        "class_pixel_counts": "",
        "class_pixel_ratios": "",
        "class_count_source": "",
        "raw_prediction_total_pixels": "",
        "visualized_prediction_total_pixels": "",
        "probability_summary": "",
        "mean_max_probability": "",
        "median_max_probability": "",
        "min_max_probability": "",
        "max_max_probability": "",
        "mask_path": "",
        "prediction_mask_raw_path": "",
        "prediction_mask_visual_path": "",
        "prediction_labels_raw_npy_path": "",
        "prediction_labels_visual_npy_path": "",
        "prediction_probabilities_npz_path": "",
        "overlay_path": "",
        "overlay_with_legend_path": "",
        "input_preview_path": "",
        "patch_inference_summary_path": "",
        "input_image_shape": "",
        "raw_prediction_shape": "",
        "resized_for_visualization": "false",
        "num_patch_warnings": 0,
        "patch_warnings": "[]",
    } | _selection_metadata_cells(selection_metadata) | _input_validation_cells(None)


def _copy_completed_outputs(
    patch_id: str,
    patch_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    outputs = patch_summary.get("outputs", {})
    copy_specs = {
        "prediction_mask_raw": (
            output_dir / "masks_raw" / f"{patch_id}__prediction_mask_raw.png"
        ),
        "prediction_mask_visual": (
            output_dir / "masks_visual" / f"{patch_id}__prediction_mask_visual.png"
        ),
        "prediction_mask": (
            output_dir / "masks" / f"{patch_id}__prediction_mask.png"
        ),
        "prediction_labels_raw_npy": (
            output_dir / "labels_raw" / f"{patch_id}__prediction_labels_raw.npy"
        ),
        "prediction_labels_visual_npy": (
            output_dir / "labels_visual" / f"{patch_id}__prediction_labels_visual.npy"
        ),
        "prediction_probabilities_npz": (
            output_dir / "labels_raw" / f"{patch_id}__prediction_probabilities.npz"
        ),
        "prediction_overlay": (
            output_dir / "overlays" / f"{patch_id}__prediction_overlay.png"
        ),
        "prediction_overlay_with_legend": (
            output_dir
            / "overlays_with_legend"
            / f"{patch_id}__prediction_overlay_with_legend.png"
        ),
        "input_preview": (
            output_dir / "input_previews" / f"{patch_id}__input_preview.png"
        ),
    }
    copied: dict[str, str] = {}
    for key, destination in copy_specs.items():
        source = outputs.get(key)
        if not source:
            copied[key] = ""
            continue
        source_path = Path(source)
        if not source_path.exists():
            copied[key] = ""
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied[key] = str(destination)
    return copied


def _result_row_from_summary(
    source_row: dict[str, str],
    *,
    patch_id: str,
    patch_summary: dict[str, Any],
    patch_summary_path: Path,
    copied_outputs: dict[str, str],
) -> dict[str, object]:
    error = patch_summary.get("error") or ""
    patch_warnings = patch_summary.get("warnings") or []
    selection_metadata = patch_summary.get("selection_metadata")
    input_validation = patch_summary.get("input_validation")
    probability_summary = patch_summary.get("probability_summary") or {}
    return {
        "patch_id": patch_id,
        "filename": source_row.get("filename", ""),
        "rank": source_row.get("rank", ""),
        "x_level0": source_row.get("x_level0", ""),
        "y_level0": source_row.get("y_level0", ""),
        "patch_size": source_row.get("patch_size", ""),
        "selection_method": source_row.get("selection_method", ""),
        "source_wsi_path": source_row.get("source_wsi_path", ""),
        "status": patch_summary.get("status", "failed"),
        "error": error,
        "resolved_device": patch_summary.get("resolved_device") or "",
        "model_name": patch_summary.get("model_name") or "",
        "prediction_shape": _json_cell(patch_summary.get("prediction_shape")),
        "visualized_mask_shape": _json_cell(patch_summary.get("visualized_mask_shape")),
        "unique_prediction_values": _json_cell(patch_summary.get("unique_prediction_values")),
        "class_pixel_counts": _json_cell(patch_summary.get("class_pixel_counts")),
        "class_pixel_ratios": _json_cell(patch_summary.get("class_pixel_ratios")),
        "class_count_source": patch_summary.get("class_count_source", ""),
        "raw_prediction_total_pixels": patch_summary.get("raw_prediction_total_pixels", ""),
        "visualized_prediction_total_pixels": patch_summary.get(
            "visualized_prediction_total_pixels",
            "",
        ),
        "probability_summary": _json_cell(probability_summary),
        "mean_max_probability": probability_summary.get("mean_max_probability", ""),
        "median_max_probability": probability_summary.get("median_max_probability", ""),
        "min_max_probability": probability_summary.get("min_max_probability", ""),
        "max_max_probability": probability_summary.get("max_max_probability", ""),
        "mask_path": copied_outputs.get("prediction_mask", ""),
        "prediction_mask_raw_path": copied_outputs.get("prediction_mask_raw", ""),
        "prediction_mask_visual_path": copied_outputs.get("prediction_mask_visual", ""),
        "prediction_labels_raw_npy_path": copied_outputs.get("prediction_labels_raw_npy", ""),
        "prediction_labels_visual_npy_path": copied_outputs.get(
            "prediction_labels_visual_npy",
            "",
        ),
        "prediction_probabilities_npz_path": copied_outputs.get(
            "prediction_probabilities_npz",
            "",
        ),
        "overlay_path": copied_outputs.get("prediction_overlay", ""),
        "overlay_with_legend_path": copied_outputs.get(
            "prediction_overlay_with_legend",
            "",
        ),
        "input_preview_path": copied_outputs.get("input_preview", ""),
        "patch_inference_summary_path": str(patch_summary_path),
        "input_image_shape": _json_cell(_shape_from_image_size(patch_summary)),
        "raw_prediction_shape": _json_cell(patch_summary.get("raw_prediction_shape")),
        "resized_for_visualization": (
            "true" if _summary_was_resized_for_visualization(patch_summary) else "false"
        ),
        "num_patch_warnings": len(patch_warnings),
        "patch_warnings": _json_cell(patch_warnings),
    } | _selection_metadata_cells(selection_metadata) | _input_validation_cells(input_validation)


def _method_config_payload(
    config: SelectedPatchSegmentationConfig,
    input_selection_dir: Path,
) -> dict[str, object]:
    return {
        "script_name": "scripts/08_segment_selected_patches.py",
        "model_name": config.model_name,
        "device": config.requested_device,
        "input_mode": config.input_mode,
        "overlay_alpha": config.overlay_alpha,
        "input_selection_dir": str(input_selection_dir),
        "limit_patches": config.limit_patches,
        "strict_input_validation": config.strict_input_validation,
        "save_probabilities": config.save_probabilities,
        "save_visual_labels_npy": config.save_visual_labels_npy,
        "created_at": _utc_now_iso(),
        "clinical_warning": CLINICAL_WARNING,
    }


def _selection_method(selection_summary: dict[str, Any], selected_rows: list[dict[str, str]]) -> str:
    selector = selection_summary.get("selector")
    if selector:
        return str(selector)
    for row in selected_rows:
        selection_method = row.get("selection_method")
        if selection_method:
            return selection_method
    return ""


def _is_true_cell(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _summarize_input_validation(result_rows: list[dict[str, object]]) -> dict[str, object]:
    completed_rows = [row for row in result_rows if row.get("input_validation_status")]
    mpp_available_rows = [
        row for row in completed_rows if _is_true_cell(row.get("mpp_available"))
    ]
    return {
        "status": "completed",
        "expected_input_width": EXPECTED_INPUT_WIDTH,
        "expected_input_height": EXPECTED_INPUT_HEIGHT,
        "expected_model_mpp": EXPECTED_MODEL_MPP,
        "mpp_tolerance": MPP_TOLERANCE,
        "num_rows_with_input_validation": len(completed_rows),
        "num_rgb_converted": sum(
            _is_true_cell(row.get("converted_to_rgb")) for row in completed_rows
        ),
        "num_input_shape_matches": sum(
            _is_true_cell(row.get("input_shape_matches_model")) for row in completed_rows
        ),
        "num_input_shape_mismatches": sum(
            row.get("input_shape_matches_model") == "false" for row in completed_rows
        ),
        "num_uint8_inputs": sum(
            _is_true_cell(row.get("is_uint8")) for row in completed_rows
        ),
        "num_valid_range_inputs": sum(
            _is_true_cell(row.get("range_looks_valid")) for row in completed_rows
        ),
        "num_mpp_available": len(mpp_available_rows),
        "num_mpp_unavailable": sum(
            row.get("mpp_available") == "false" for row in completed_rows
        ),
        "num_mpp_within_tolerance": sum(
            _is_true_cell(row.get("mpp_within_tolerance")) for row in completed_rows
        ),
        "num_mpp_outside_tolerance": sum(
            row.get("mpp_available") == "true"
            and row.get("mpp_within_tolerance") == "false"
            for row in completed_rows
        ),
        "clinical_warning": CLINICAL_WARNING,
    }


def segment_selected_patches(config: SelectedPatchSegmentationConfig) -> dict[str, Any]:
    """Segment selected patch PNGs and write per-patch and run-level manifests."""
    start_time = time.perf_counter()
    root_dir = config.root_dir.resolve()
    input_selection_dir = _resolve_path(config.input_selection_dir, root_dir)
    output_dir = _resolve_path(config.output_dir, root_dir)
    if config.limit_patches is not None and config.limit_patches <= 0:
        raise ValueError("--limit-patches must be positive when provided.")
    if not 0 <= config.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must be between 0 and 1.")

    _validate_input_selection_dir(input_selection_dir)
    _prepare_output_dir(output_dir=output_dir, root_dir=root_dir, overwrite=config.overwrite)

    selected_metadata_path = input_selection_dir / "selected_metadata.csv"
    selection_summary_path = input_selection_dir / "selection_summary.json"
    method_config_path = input_selection_dir / "method_config.json"
    selected_dir = input_selection_dir / "selected"
    per_patch_csv_path = output_dir / "per_patch_segmentation.csv"
    global_summary_path = output_dir / "inference_summary.json"
    run_method_config_path = output_dir / "method_config.json"
    input_validation_summary_path = output_dir / "input_validation_summary.json"

    selected_rows = _read_csv(selected_metadata_path)
    selection_summary = _read_json(selection_summary_path)
    _read_json(method_config_path)
    _write_json(
        _method_config_payload(config, input_selection_dir=input_selection_dir),
        run_method_config_path,
    )

    warnings: list[str] = []
    result_rows: list[dict[str, object]] = []
    attempted = 0
    completed = 0
    failed = 0
    skipped = 0
    valid_seen = 0
    num_patch_warnings = 0
    num_patches_with_warnings = 0
    num_patches_with_resized_visualization = 0
    num_patches_with_probability_summary = 0
    unique_patch_warnings: set[str] = set()

    for index, row in enumerate(selected_rows, start=1):
        selection_metadata = build_selection_metadata_for_patch(
            selected_row=row,
            selection_summary=selection_summary,
            input_selection_dir=input_selection_dir,
            selected_metadata_path=selected_metadata_path,
            selection_summary_path=selection_summary_path,
        )
        metadata_warnings = [
            str(warning) for warning in selection_metadata.get("metadata_warnings", [])
        ]
        for warning in metadata_warnings:
            warnings.append(f"Row {index} metadata warning: {warning}")

        filename = row.get("filename", "").strip()
        if not filename:
            skipped += 1
            warnings.append(f"Row {index} skipped: missing filename.")
            result_rows.append(
                _empty_result_row(
                    row,
                    patch_id="",
                    status="skipped",
                    error="missing filename",
                    selection_metadata=selection_metadata,
                )
            )
            continue

        patch_path = selected_dir / filename
        patch_id = _patch_id_for_row(row)
        if not patch_path.exists():
            skipped += 1
            warnings.append(f"Row {index} skipped: selected patch file not found: {patch_path}")
            result_rows.append(
                _empty_result_row(
                    row,
                    patch_id=patch_id,
                    status="skipped",
                    error=f"selected patch file not found: {patch_path}",
                    selection_metadata=selection_metadata,
                )
            )
            continue

        if config.limit_patches is not None and valid_seen >= config.limit_patches:
            continue
        valid_seen += 1
        attempted += 1

        patch_output_dir = output_dir / "per_patch" / patch_id
        try:
            from src.inference.tiatoolbox_inference import run_inference_smoke_test

            patch_summary, patch_summary_path = run_inference_smoke_test(
                image_path=patch_path,
                output_dir=patch_output_dir,
                root_dir=root_dir,
                model_name=config.model_name,
                requested_device=config.requested_device,
                input_mode=config.input_mode,
                overlay_alpha=config.overlay_alpha,
                clear_output=True,
                strict_input_validation=config.strict_input_validation,
                selection_metadata=selection_metadata,
                save_probabilities=config.save_probabilities,
                save_visual_labels_npy=config.save_visual_labels_npy,
            )
            patch_warnings = [str(warning) for warning in patch_summary.get("warnings") or []]
            if patch_warnings:
                num_patches_with_warnings += 1
                num_patch_warnings += len(patch_warnings)
                unique_patch_warnings.update(patch_warnings)
            if _summary_was_resized_for_visualization(patch_summary):
                num_patches_with_resized_visualization += 1
            if (patch_summary.get("probability_summary") or {}).get("available") is True:
                num_patches_with_probability_summary += 1
            copied_outputs = (
                _copy_completed_outputs(patch_id, patch_summary, output_dir)
                if patch_summary.get("status") == "completed"
                else {}
            )
            result_rows.append(
                _result_row_from_summary(
                    row,
                    patch_id=patch_id,
                    patch_summary=patch_summary,
                    patch_summary_path=patch_summary_path,
                    copied_outputs=copied_outputs,
                )
            )
            if patch_summary.get("status") == "completed":
                completed += 1
            else:
                failed += 1
                warnings.append(
                    f"Patch {patch_id} failed: {patch_summary.get('error') or 'unknown error'}"
                )
        except Exception as exc:  # noqa: BLE001 - continue per patch
            failed += 1
            patch_summary_path = patch_output_dir / "inference_summary.json"
            result_rows.append(
                _empty_result_row(
                    row,
                    patch_id=patch_id,
                    status="failed",
                    error=str(exc),
                    selection_metadata=selection_metadata,
                )
                | {"patch_inference_summary_path": str(patch_summary_path)}
            )
            warnings.append(f"Patch {patch_id} failed: {exc}")

    _write_csv(result_rows, per_patch_csv_path, PER_PATCH_SEGMENTATION_FIELDS)
    input_validation_summary = _summarize_input_validation(result_rows)
    _write_json(input_validation_summary, input_validation_summary_path)

    if num_patch_warnings:
        warnings.append(
            f"Patch-level warnings observed: {num_patch_warnings} warnings "
            f"across {num_patches_with_warnings} patches."
        )

    if failed or skipped or warnings or num_patch_warnings:
        status = "completed_with_warnings"
    else:
        status = "completed"

    summary = {
        "status": status,
        "input_selection_dir": str(input_selection_dir),
        "output_dir": str(output_dir),
        "selector": selection_summary.get("selector", ""),
        "selection_method": _selection_method(selection_summary, selected_rows),
        "model_name": config.model_name,
        "requested_device": config.requested_device,
        "input_mode": config.input_mode,
        "overlay_alpha": config.overlay_alpha,
        "num_metadata_rows": len(selected_rows),
        "num_patches_attempted": attempted,
        "num_patches_completed": completed,
        "num_patches_failed": failed,
        "num_patches_skipped": skipped,
        "num_patch_warnings": num_patch_warnings,
        "num_patches_with_warnings": num_patches_with_warnings,
        "unique_patch_warnings": sorted(unique_patch_warnings),
        "num_patches_with_resized_visualization": num_patches_with_resized_visualization,
        "num_patches_with_probability_summary": num_patches_with_probability_summary,
        "strict_input_validation": config.strict_input_validation,
        "save_probabilities": config.save_probabilities,
        "save_visual_labels_npy": config.save_visual_labels_npy,
        "prediction_resolution_note": PREDICTION_RESOLUTION_NOTE,
        "input_validation_summary": input_validation_summary,
        "input_validation_summary_json": str(input_validation_summary_path),
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
        "selection_summary_path": str(selection_summary_path),
        "selection_method_config_path": str(method_config_path),
        "per_patch_segmentation_csv": str(per_patch_csv_path),
        "warnings": warnings,
        "clinical_warning": CLINICAL_WARNING,
    }
    _write_json(summary, global_summary_path)
    return summary
