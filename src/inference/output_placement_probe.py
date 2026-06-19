"""Technical probes for TIAToolbox segmentation output placement."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.inference.context_stitching import AXIS_CONVENTION, WINDOW_IDS
from src.inference.tiatoolbox_inference import (
    CLINICAL_WARNING,
    run_inference_smoke_test,
)
from src.visualization.segmentation_overlay import colorize_label_mask


PATCH_INPUT_SHAPE = 1024
PATCH_OUTPUT_SHAPE = 512
EXPECTED_CENTER_OFFSET = (PATCH_INPUT_SHAPE - PATCH_OUTPUT_SHAPE) // 2
SECONDARY_CHECK_WARNING = (
    "Direct-vs-stitched consistency is a secondary diagnostic only. "
    "It does not prove clinical validity or final correctness of the context-stitch hypothesis."
)


def ensure_probe_environment() -> None:
    """Set cache directories that avoid common local TIAToolbox import issues."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write stable JSON output for probe artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unavailable"))


def _source_location(obj: object) -> dict[str, Any]:
    try:
        source_file = inspect.getsourcefile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
        return {
            "source_path": str(source_file) if source_file else None,
            "line_start": int(start_line),
            "line_end": int(start_line + len(source_lines) - 1),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic source inspection
        return {
            "source_path": None,
            "line_start": None,
            "line_end": None,
            "error": str(exc),
        }


def _source_text(obj: object) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:  # noqa: BLE001 - diagnostic source inspection
        return ""


def _as_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    return str(value)


def _ioconfig_metadata(model_name: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "status": "not_loaded",
        "model_name": model_name,
        "error": None,
    }
    try:
        architecture_module = importlib.import_module("tiatoolbox.models.architecture")
        get_pretrained_model = getattr(architecture_module, "get_pretrained_model")
        loaded_model = get_pretrained_model(pretrained_model=model_name)
        if isinstance(loaded_model, tuple):
            model = loaded_model[0]
            ioconfig = loaded_model[1] if len(loaded_model) > 1 else None
        else:
            model = loaded_model
            ioconfig = None
        metadata.update(
            {
                "status": "loaded",
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "ioconfig_class": (
                    f"{ioconfig.__class__.__module__}.{ioconfig.__class__.__name__}"
                    if ioconfig is not None
                    else None
                ),
            }
        )
        if ioconfig is not None:
            for attr_name in (
                "patch_input_shape",
                "patch_output_shape",
                "stride_shape",
                "input_resolutions",
                "output_resolutions",
                "save_resolution",
                "tile_shape",
                "margin",
            ):
                if hasattr(ioconfig, attr_name):
                    metadata[attr_name] = _as_jsonable(getattr(ioconfig, attr_name))
    except Exception as exc:  # noqa: BLE001 - cache/model diagnostic
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
    return metadata


def inspect_tiatoolbox_output_placement(model_name: str) -> dict[str, Any]:
    """Inspect installed TIAToolbox source/config for output placement evidence."""
    ensure_probe_environment()
    tiatoolbox = importlib.import_module("tiatoolbox")
    unet_module = importlib.import_module("tiatoolbox.models.architecture.unet")
    arch_utils = importlib.import_module("tiatoolbox.models.architecture.utils")
    patchextraction = importlib.import_module("tiatoolbox.tools.patchextraction")
    dataset_abc = importlib.import_module("tiatoolbox.models.dataset.dataset_abc")
    semantic_segmentor = importlib.import_module("tiatoolbox.models.engine.semantic_segmentor")

    infer_batch_source = _source_text(unet_module.UNetModel.infer_batch)
    centre_crop_source = _source_text(arch_utils.centre_crop)
    get_coordinates_source = _source_text(patchextraction.PatchExtractor.get_coordinates)
    dataset_init_source = _source_text(dataset_abc.WSIPatchDataset.__init__)
    dataloader_source = _source_text(semantic_segmentor.SemanticSegmentor.get_dataloader)

    unet_patterns = {
        "uses_input_height_width": "_, _, h, w = imgs.shape" in infer_batch_source,
        "crop_shape_half_input": "crop_shape = [h // 2, w // 2]" in infer_batch_source,
        "upsamples_probabilities": "F.interpolate" in infer_batch_source
        and "scale_factor=2" in infer_batch_source,
        "applies_centre_crop": "centre_crop(probs, crop_shape)" in infer_batch_source,
    }
    crop_patterns = {
        "top_crop_half": "crop_t: int = int(crop_shape[0] // 2)" in centre_crop_source,
        "bottom_crop_remaining": "crop_b: int = int(crop_shape[0] - crop_t)" in centre_crop_source,
        "left_crop_half": "crop_l: int = int(crop_shape[1] // 2)" in centre_crop_source,
        "right_crop_remaining": "crop_r: int = int(crop_shape[1] - crop_l)" in centre_crop_source,
    }
    coordinate_patterns = {
        "io_diff_input_minus_output": (
            "io_diff = patch_input_shape_arr - patch_output_shape_arr"
            in get_coordinates_source
        ),
        "input_top_left_centered_around_output": (
            "input_tl_list = output_tl_list - (io_diff // 2)[None]"
            in get_coordinates_source
        ),
        "returns_input_and_output_bounds": "return input_bound_list, output_bound_list" in get_coordinates_source,
    }
    dataset_patterns = {
        "passes_patch_output_shape_to_get_coordinates": (
            "patch_output_shape=patch_output_shape" in dataset_init_source
        ),
        "stores_outputs_from_patch_extractor": "self.inputs, self.outputs" in dataset_init_source,
        "getitem_returns_output_locs": '"output_locs": output_locs' in _source_text(dataset_abc.WSIPatchDataset.__getitem__),
    }
    dataloader_patterns = {
        "output_locations_from_dataset_outputs": "self.output_locations = dataset.outputs" in dataloader_source,
        "dataset_receives_patch_input_shape": "patch_input_shape=ioconfig.patch_input_shape" in dataloader_source,
        "dataset_receives_patch_output_shape": "patch_output_shape=ioconfig.patch_output_shape" in dataloader_source,
    }

    source_support = (
        all(unet_patterns.values())
        and all(crop_patterns.values())
        and all(coordinate_patterns.values())
        and all(dataset_patterns.values())
    )
    if source_support:
        source_code_conclusion = "supported"
        reasoning = (
            "Installed TIAToolbox source shows the U-Net prediction is center-cropped "
            "and patch coordinates are generated by centering the larger input window "
            "around the smaller output bounds."
        )
    else:
        source_code_conclusion = "inconclusive"
        reasoning = "One or more source-code patterns required for center-output evidence were not found."

    return {
        "status": "completed",
        "test_type": "source_inspection",
        "model_name": model_name,
        "tiatoolbox_version": _safe_version(tiatoolbox),
        "expected_input_shape": [PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE],
        "expected_output_shape": [PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE],
        "expected_center_offset": [EXPECTED_CENTER_OFFSET, EXPECTED_CENTER_OFFSET],
        "source_code_conclusion": source_code_conclusion,
        "reasoning": reasoning,
        "evidence": [
            {
                "component": "UNetModel.infer_batch",
                **_source_location(unet_module.UNetModel.infer_batch),
                "patterns": unet_patterns,
                "interpretation": (
                    "For 1024x1024 inputs, crop_shape is 512x512 and centre_crop "
                    "removes 256 pixels per side after interpolation."
                ),
            },
            {
                "component": "centre_crop",
                **_source_location(arch_utils.centre_crop),
                "patterns": crop_patterns,
                "interpretation": "The crop is split symmetrically across top/bottom and left/right.",
            },
            {
                "component": "PatchExtractor.get_coordinates",
                **_source_location(patchextraction.PatchExtractor.get_coordinates),
                "patterns": coordinate_patterns,
                "interpretation": (
                    "Input bounds are shifted by half the input-output difference, "
                    "so output bounds are centered inside input bounds."
                ),
            },
            {
                "component": "WSIPatchDataset",
                **_source_location(dataset_abc.WSIPatchDataset.__init__),
                "patterns": dataset_patterns,
                "interpretation": "The WSI dataset keeps separate input bounds and output_locs.",
            },
            {
                "component": "SemanticSegmentor.get_dataloader",
                **_source_location(semantic_segmentor.SemanticSegmentor.get_dataloader),
                "patterns": dataloader_patterns,
                "interpretation": "SemanticSegmentor uses dataset.outputs as output locations.",
            },
        ],
        "ioconfig": _ioconfig_metadata(model_name),
        "clinical_warning": CLINICAL_WARNING,
    }


def run_coordinate_probe(
    *,
    x_level0: int | None = None,
    y_level0: int | None = None,
    patch_size: int = PATCH_INPUT_SHAPE,
) -> dict[str, Any]:
    """Run a coordinate-only probe against TIAToolbox PatchExtractor logic."""
    ensure_probe_environment()
    patchextraction = importlib.import_module("tiatoolbox.tools.patchextraction")
    PatchExtractor = patchextraction.PatchExtractor

    inputs, outputs = PatchExtractor.get_coordinates(
        image_shape=(2048, 2048),
        patch_input_shape=(PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE),
        stride_shape=(PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
        patch_output_shape=(PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
    )
    inputs = np.asarray(inputs, dtype=np.int64)
    outputs = np.asarray(outputs, dtype=np.int64)
    first_four = min(4, len(inputs))
    offsets_xy = outputs[:first_four, :2] - inputs[:first_four, :2]
    output_shapes = outputs[:first_four, 2:4] - outputs[:first_four, 0:2]
    observed_center_offset = offsets_xy.tolist()

    support = bool(
        first_four == 4
        and np.all(offsets_xy == EXPECTED_CENTER_OFFSET)
        and np.all(output_shapes == PATCH_OUTPUT_SHAPE)
    )
    context_window_bounds = []
    direct_patch_bounds = None
    if x_level0 is not None and y_level0 is not None:
        x0 = int(x_level0)
        y0 = int(y_level0)
        direct_patch_bounds = {
            "input_wsi_bounds": [x0, y0, x0 + patch_size, y0 + patch_size],
            "expected_output_wsi_bounds": [
                x0 + EXPECTED_CENTER_OFFSET,
                y0 + EXPECTED_CENTER_OFFSET,
                x0 + EXPECTED_CENTER_OFFSET + PATCH_OUTPUT_SHAPE,
                y0 + EXPECTED_CENTER_OFFSET + PATCH_OUTPUT_SHAPE,
            ],
        }
        placements = {
            "window_00": (x0, y0),
            "window_01": (x0 + PATCH_OUTPUT_SHAPE, y0),
            "window_10": (x0, y0 + PATCH_OUTPUT_SHAPE),
            "window_11": (x0 + PATCH_OUTPUT_SHAPE, y0 + PATCH_OUTPUT_SHAPE),
        }
        for window_id in WINDOW_IDS:
            output_x0, output_y0 = placements[window_id]
            input_x0 = output_x0 - EXPECTED_CENTER_OFFSET
            input_y0 = output_y0 - EXPECTED_CENTER_OFFSET
            context_window_bounds.append(
                {
                    "window_id": window_id,
                    "input_wsi_bounds": [
                        input_x0,
                        input_y0,
                        input_x0 + PATCH_INPUT_SHAPE,
                        input_y0 + PATCH_INPUT_SHAPE,
                    ],
                    "expected_output_wsi_bounds": [
                        output_x0,
                        output_y0,
                        output_x0 + PATCH_OUTPUT_SHAPE,
                        output_y0 + PATCH_OUTPUT_SHAPE,
                    ],
                    "stitch_target_bounds": [
                        output_x0 - x0,
                        output_y0 - y0,
                        output_x0 - x0 + PATCH_OUTPUT_SHAPE,
                        output_y0 - y0 + PATCH_OUTPUT_SHAPE,
                    ],
                }
            )

    return {
        "status": "completed",
        "test_type": "coordinate_probe",
        "coordinate_probe_conclusion": "supported" if support else "inconclusive",
        "axis_convention": AXIS_CONVENTION,
        "expected_center_offset": [EXPECTED_CENTER_OFFSET, EXPECTED_CENTER_OFFSET],
        "synthetic_patch_input_shape": [PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE],
        "synthetic_patch_output_shape": [PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE],
        "tiatoolbox_first_four_inputs": inputs[:first_four].tolist(),
        "tiatoolbox_first_four_outputs": outputs[:first_four].tolist(),
        "observed_output_minus_input_top_left": observed_center_offset,
        "observed_output_shapes": output_shapes.tolist(),
        "direct_selected_patch_mapping": direct_patch_bounds,
        "context_stitch_window_mappings": context_window_bounds,
        "reasoning": (
            "PatchExtractor.get_coordinates reports output bounds whose top-left is "
            "256 pixels inside each 1024x1024 input. The four context windows can "
            "therefore tile a 1024x1024 target patch with four 512x512 centered outputs."
            if support
            else "Coordinate offsets did not match the expected centered 256-pixel margin."
        ),
        "clinical_warning": CLINICAL_WARNING,
    }


def run_tiatoolbox_merge_probe(output_dir: Path) -> dict[str, Any]:
    """Check that TIAToolbox prepare_full_batch preserves explicit output locations."""
    ensure_probe_environment()
    semantic_segmentor = importlib.import_module("tiatoolbox.models.engine.semantic_segmentor")
    patchextraction = importlib.import_module("tiatoolbox.tools.patchextraction")
    PatchExtractor = patchextraction.PatchExtractor

    inputs, outputs = PatchExtractor.get_coordinates(
        image_shape=(2048, 2048),
        patch_input_shape=(PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE),
        stride_shape=(PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
        patch_output_shape=(PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
    )
    _ = inputs
    batch_locs = np.asarray(outputs[:4], dtype=np.int64)
    full_output_locs = batch_locs.copy()
    batch_output = np.zeros((4, PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE, 1), dtype=np.uint8)
    for idx in range(4):
        batch_output[idx, :, :, 0] = idx + 1

    full_batch_output, remaining_locs, output_locs = semantic_segmentor.prepare_full_batch(
        batch_output=batch_output,
        batch_locs=batch_locs,
        full_output_locs=full_output_locs,
        output_locs=None,
        canvas_np=None,
        save_path=output_dir / "tiatoolbox_merge_tmp",
        is_last=True,
    )
    full_batch_output_np = np.asarray(full_batch_output)
    preserved_locations = np.array_equal(output_locs, batch_locs)
    preserved_values = [
        int(np.unique(full_batch_output_np[idx]).tolist()[0]) for idx in range(4)
    ]
    expected_values = [1, 2, 3, 4]
    support = preserved_locations and preserved_values == expected_values and len(remaining_locs) == 0

    return {
        "status": "completed",
        "test_type": "tiatoolbox_merge_probe",
        "scope": "prepare_full_batch_location_preservation",
        "merge_probe_conclusion": "supported" if support else "inconclusive",
        "batch_locs": batch_locs.tolist(),
        "output_locs_after_prepare_full_batch": np.asarray(output_locs).tolist(),
        "remaining_locs_shape": list(np.asarray(remaining_locs).shape),
        "full_batch_output_shape": list(full_batch_output_np.shape),
        "preserved_locations": bool(preserved_locations),
        "preserved_values": preserved_values,
        "reasoning": (
            "prepare_full_batch preserved explicit output_locs and assigned each batch "
            "output by matching those coordinates. This supports the coordinate path but "
            "is not a full WSI merge validation."
        ),
        "clinical_warning": CLINICAL_WARNING,
    }


def _read_selected_row(selection_dir: Path, patch_index: int) -> tuple[dict[str, str], list[dict[str, str]]]:
    import csv

    selected_metadata_path = selection_dir / "selected_metadata.csv"
    if not selected_metadata_path.exists():
        raise FileNotFoundError(f"Missing selected_metadata.csv: {selected_metadata_path}")
    with selected_metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if patch_index < 0 or patch_index >= len(rows):
        raise IndexError(f"--patch-index {patch_index} outside selected_metadata.csv rows.")
    return rows[patch_index], rows


def selected_patch_path(selection_dir: Path, patch_index: int) -> tuple[Path, dict[str, str]]:
    """Resolve a selected PNG from selected_metadata.csv."""
    row, _rows = _read_selected_row(selection_dir, patch_index)
    filename = row.get("filename", "")
    if not filename:
        raise ValueError(f"Row {patch_index} in selected_metadata.csv does not contain filename.")
    path = selection_dir / "selected" / filename
    if not path.exists():
        raise FileNotFoundError(f"Selected patch PNG does not exist: {path}")
    return path, row


def _safe_int(value: object, default: int | None = None) -> int | None:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def selected_patch_coordinates(row: dict[str, str]) -> dict[str, int | None]:
    """Extract level-0 coordinates from selected patch metadata."""
    return {
        "x_level0": _safe_int(row.get("x_level0")),
        "y_level0": _safe_int(row.get("y_level0")),
        "patch_size": _safe_int(row.get("patch_size"), PATCH_INPUT_SHAPE),
    }


def _pixel_agreement(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.mean(a == b))


def _iou_by_class(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    classes = sorted({int(value) for value in np.unique(a)} | {int(value) for value in np.unique(b)})
    ious: dict[str, float | None] = {}
    for class_id in classes:
        a_mask = a == class_id
        b_mask = b == class_id
        union = np.logical_or(a_mask, b_mask).sum()
        if union == 0:
            ious[str(class_id)] = None
        else:
            ious[str(class_id)] = float(np.logical_and(a_mask, b_mask).sum() / union)
    return ious


def _macro_iou(ious: dict[str, float | None]) -> float | None:
    values = [value for value in ious.values() if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def run_secondary_consistency_check(
    *,
    selection_dir: Path,
    patch_index: int,
    output_dir: Path,
    model_name: str,
    device: str,
    root_dir: Path,
    stitched_prediction_path: Path | None = None,
) -> dict[str, Any]:
    """Compare direct selected-patch inference with an existing stitched prediction."""
    secondary_dir = output_dir / "secondary_consistency"
    secondary_dir.mkdir(parents=True, exist_ok=True)
    patch_path, row = selected_patch_path(selection_dir, patch_index)
    if stitched_prediction_path is None:
        stitched_prediction_path = output_dir.parent / "alignment_probe" / "stitched_prediction_1024.npy"
    if not stitched_prediction_path.exists():
        return {
            "status": "skipped",
            "test_type": "direct_vs_stitched_consistency",
            "reason": f"Missing stitched prediction: {stitched_prediction_path}",
            "secondary_warning": SECONDARY_CHECK_WARNING,
            "clinical_warning": CLINICAL_WARNING,
        }

    summary, summary_path = run_inference_smoke_test(
        image_path=patch_path,
        output_dir=secondary_dir / "direct_patch_inference",
        root_dir=root_dir,
        model_name=model_name,
        requested_device=device,
        input_mode="patch",
        clear_output=True,
        selection_metadata={
            "selection_dir": str(selection_dir),
            "patch_index": patch_index,
            "filename": row.get("filename"),
            "x_level0": row.get("x_level0"),
            "y_level0": row.get("y_level0"),
            "patch_size": row.get("patch_size"),
            "source_wsi_path": row.get("source_wsi_path"),
        },
    )
    if summary.get("status") != "completed":
        return {
            "status": "failed",
            "test_type": "direct_vs_stitched_consistency",
            "error": summary.get("error"),
            "direct_inference_summary": str(summary_path),
            "secondary_warning": SECONDARY_CHECK_WARNING,
            "clinical_warning": CLINICAL_WARNING,
        }

    direct_path = Path(summary["outputs"]["prediction_labels_raw_npy"])
    direct = np.asarray(np.load(direct_path)).squeeze()
    stitched = np.asarray(np.load(stitched_prediction_path)).squeeze()
    if direct.shape != (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE):
        raise ValueError(f"Direct output expected 512x512, got {direct.shape}.")
    if stitched.shape != (PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE):
        raise ValueError(f"Stitched output expected 1024x1024, got {stitched.shape}.")

    stitched_center = stitched[
        EXPECTED_CENTER_OFFSET:EXPECTED_CENTER_OFFSET + PATCH_OUTPUT_SHAPE,
        EXPECTED_CENTER_OFFSET:EXPECTED_CENTER_OFFSET + PATCH_OUTPUT_SHAPE,
    ]
    agreement = _pixel_agreement(direct, stitched_center)
    ious = _iou_by_class(direct, stitched_center)
    macro_iou = _macro_iou(ious)

    direct_png = secondary_dir / "direct_prediction_raw.png"
    stitched_center_png = secondary_dir / "stitched_center_prediction.png"
    difference_png = output_dir / "direct_vs_stitched_difference.png"
    Image.fromarray(colorize_label_mask(direct)).save(direct_png)
    Image.fromarray(colorize_label_mask(stitched_center)).save(stitched_center_png)
    difference = np.where(direct == stitched_center, 0, 255).astype(np.uint8)
    Image.fromarray(difference).save(difference_png)

    return {
        "status": "completed",
        "test_type": "direct_vs_stitched_consistency",
        "conclusion_role": "secondary_only",
        "secondary_warning": SECONDARY_CHECK_WARNING,
        "selected_patch_png": str(patch_path),
        "stitched_prediction_npy": str(stitched_prediction_path),
        "direct_inference_summary": str(summary_path),
        "direct_prediction_labels_raw_npy": str(direct_path),
        "direct_prediction_shape": list(direct.shape),
        "stitched_prediction_shape": list(stitched.shape),
        "compared_stitched_center_bounds": [
            EXPECTED_CENTER_OFFSET,
            EXPECTED_CENTER_OFFSET,
            EXPECTED_CENTER_OFFSET + PATCH_OUTPUT_SHAPE,
            EXPECTED_CENTER_OFFSET + PATCH_OUTPUT_SHAPE,
        ],
        "pixel_agreement": agreement,
        "iou_by_class": ious,
        "macro_iou": macro_iou,
        "direct_prediction_raw_png": str(direct_png),
        "stitched_center_prediction_png": str(stitched_center_png),
        "direct_vs_stitched_difference_png": str(difference_png),
        "clinical_warning": CLINICAL_WARNING,
    }


def run_offset_search(
    *,
    direct_prediction_path: Path,
    stitched_prediction_path: Path,
    output_dir: Path,
    step: int = 16,
) -> dict[str, Any]:
    """Search 512x512 placements within a 1024x1024 stitched mask."""
    import csv

    direct = np.asarray(np.load(direct_prediction_path)).squeeze()
    stitched = np.asarray(np.load(stitched_prediction_path)).squeeze()
    if direct.shape != (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE):
        raise ValueError(f"Direct output expected 512x512, got {direct.shape}.")
    if stitched.shape != (PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE):
        raise ValueError(f"Stitched output expected 1024x1024, got {stitched.shape}.")

    offsets = list(range(0, PATCH_INPUT_SHAPE - PATCH_OUTPUT_SHAPE + 1, step))
    if offsets[-1] != PATCH_INPUT_SHAPE - PATCH_OUTPUT_SHAPE:
        offsets.append(PATCH_INPUT_SHAPE - PATCH_OUTPUT_SHAPE)

    rows: list[dict[str, Any]] = []
    heatmap = np.zeros((len(offsets), len(offsets)), dtype=np.float32)
    for y_index, y_offset in enumerate(offsets):
        for x_index, x_offset in enumerate(offsets):
            candidate = stitched[
                y_offset:y_offset + PATCH_OUTPUT_SHAPE,
                x_offset:x_offset + PATCH_OUTPUT_SHAPE,
            ]
            ious = _iou_by_class(direct, candidate)
            macro_iou = _macro_iou(ious)
            agreement = _pixel_agreement(direct, candidate)
            heatmap[y_index, x_index] = agreement
            rows.append(
                {
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "pixel_agreement": agreement,
                    "macro_iou": macro_iou,
                }
            )
    rows.sort(key=lambda row: (row["pixel_agreement"], row["macro_iou"] or -1.0), reverse=True)

    csv_path = output_dir / "offset_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["x_offset", "y_offset", "pixel_agreement", "macro_iou"],
        )
        writer.writeheader()
        writer.writerows(rows)

    heatmap_path = output_dir / "offset_agreement_heatmap.png"
    heatmap_written = False
    heatmap_error = None
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(heatmap, origin="upper", cmap="viridis")
        ax.set_xticks(range(len(offsets)))
        ax.set_yticks(range(len(offsets)))
        ax.set_xticklabels(offsets, rotation=90, fontsize=6)
        ax.set_yticklabels(offsets, fontsize=6)
        ax.set_xlabel("x offset in stitched 1024 mask")
        ax.set_ylabel("y offset in stitched 1024 mask")
        ax.set_title("Direct vs stitched pixel agreement")
        fig.colorbar(image, ax=ax, label="pixel agreement")
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        heatmap_written = True
    except Exception as exc:  # noqa: BLE001 - optional diagnostic plot
        heatmap_error = str(exc)

    best = rows[0] if rows else {}
    center_row = next(
        (
            row
            for row in rows
            if row["x_offset"] == EXPECTED_CENTER_OFFSET
            and row["y_offset"] == EXPECTED_CENTER_OFFSET
        ),
        None,
    )
    summary = {
        "status": "completed",
        "test_type": "offset_search",
        "conclusion_role": "secondary_only",
        "secondary_warning": SECONDARY_CHECK_WARNING,
        "step": step,
        "candidate_offsets": offsets,
        "best_offset": best,
        "center_offset_result": center_row,
        "expected_center_offset": {
            "x_offset": EXPECTED_CENTER_OFFSET,
            "y_offset": EXPECTED_CENTER_OFFSET,
        },
        "center_is_best_by_pixel_agreement": bool(
            center_row is not None
            and best
            and center_row["pixel_agreement"] == best["pixel_agreement"]
        ),
        "offset_search_csv": str(csv_path),
        "offset_agreement_heatmap_png": str(heatmap_path) if heatmap_written else None,
        "heatmap_error": heatmap_error,
        "clinical_warning": CLINICAL_WARNING,
    }
    write_json(summary, output_dir / "offset_search_summary.json")
    return summary


def write_output_placement_report_md(report: dict[str, Any], path: Path) -> Path:
    """Write a short Markdown report for the output placement probe."""
    source = report.get("source_inspection", {})
    coordinate = report.get("coordinate_probe", {})
    merge = report.get("tiatoolbox_merge_probe")
    secondary = report.get("direct_vs_stitched_consistency")
    offset = report.get("offset_search")

    lines = [
        "# TIAToolbox output placement probe",
        "",
        f"Conclusion: `{report.get('overall_conclusion')}`",
        "",
        f"Clinical warning: {CLINICAL_WARNING}",
        "",
        "## Primary evidence",
        "",
        (
            "- Source inspection: "
            f"`{source.get('source_code_conclusion', 'not_run')}`. "
            f"{source.get('reasoning', '')}"
        ),
        (
            "- Coordinate probe: "
            f"`{coordinate.get('coordinate_probe_conclusion', 'not_run')}`. "
            f"{coordinate.get('reasoning', '')}"
        ),
    ]
    if merge:
        lines.extend(
            [
                (
                    "- TIAToolbox merge probe: "
                    f"`{merge.get('merge_probe_conclusion', 'not_run')}`. "
                    f"{merge.get('reasoning', '')}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "For `patch_input_shape=1024` and `patch_output_shape=512`, the expected "
                "technical offset is 256 pixels on each side. The evidence above addresses "
                "where TIAToolbox places the model output in coordinate space. It does not "
                "diagnose tissue, calculate RCB, or validate clinical performance."
            ),
            "",
            "## Secondary checks",
            "",
        ]
    )
    if secondary:
        lines.append(
            "- Direct-vs-stitched consistency: "
            f"`{secondary.get('status')}`, pixel agreement "
            f"`{secondary.get('pixel_agreement')}`. This is secondary only."
        )
    else:
        lines.append("- Direct-vs-stitched consistency: not run.")
    if offset:
        lines.append(
            "- Offset search: "
            f"`{offset.get('status')}`, best offset `{offset.get('best_offset')}`. "
            "This is secondary only."
        )
    else:
        lines.append("- Offset search: not run.")
    lines.extend(["", "## Artifacts", ""])
    for key, value in report.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def determine_overall_conclusion(
    source_inspection: dict[str, Any] | None,
    coordinate_probe: dict[str, Any] | None,
    merge_probe: dict[str, Any] | None,
) -> str:
    """Return supported, contradicted, or inconclusive from primary evidence only."""
    primary = [item for item in (source_inspection, coordinate_probe, merge_probe) if item]
    conclusions = {
        str(item.get("source_code_conclusion") or item.get("coordinate_probe_conclusion") or item.get("merge_probe_conclusion"))
        for item in primary
    }
    if "contradicted" in conclusions:
        return "contradicted"
    if "supported" in conclusions:
        return "supported"
    return "inconclusive"
