"""Compare no-overlap and overlap-aware context stitching strategies."""

from __future__ import annotations

import csv
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.inference.context_stitching import (
    AXIS_CONVENTION,
    CLINICAL_WARNING,
    WINDOW_IDS,
    compute_context_geometry,
    extract_windows_2x2,
)
from src.inference.tiatoolbox_inference import (
    _build_segmentor,
    _prediction_from_run_output,
    _probability_array_for_label_mask,
    _probability_from_run_output,
)
from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, resolve_torch_device
from src.visualization.segmentation_overlay import colorize_label_mask, overlay_label_mask


PATCH_INPUT_SHAPE = 1024
PATCH_OUTPUT_SHAPE = 512
PATCH_SIZE = 1024
CENTER_MARGIN = (PATCH_INPUT_SHAPE - PATCH_OUTPUT_SHAPE) // 2
PADDING_RGB = (255, 255, 255)
AGREEMENT_METRIC_WARNING = (
    "These are agreement metrics between inference strategies, not accuracy metrics "
    "against ground truth."
)


@dataclass(frozen=True)
class StrategyComparisonConfig:
    """Configuration for the isolated context-stitch comparison probe."""

    selection_dir: Path
    output_dir: Path
    patch_indices: tuple[int, ...]
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "cpu"
    overlap_stride: int = 450
    blend_mode: str = "uniform"
    run_no_overlap: bool = True
    run_overlap_aware: bool = True
    overlay_alpha: float = 0.45
    seam_band: int = 8


def ensure_runtime_environment() -> None:
    """Set cache directories used by TIAToolbox/Matplotlib when not provided."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _first_non_empty(*values: object) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""


def _safe_int(value: object, default: int | None = None) -> int | None:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def load_selected_patch_metadata(selection_dir: Path, patch_index: int) -> dict[str, Any]:
    """Load one selected patch row and resolve WSI-level metadata."""
    selected_metadata_path = selection_dir / "selected_metadata.csv"
    selection_summary_path = selection_dir / "selection_summary.json"
    if not selected_metadata_path.exists():
        raise FileNotFoundError(f"Missing selected_metadata.csv: {selected_metadata_path}")
    if not selection_summary_path.exists():
        raise FileNotFoundError(f"Missing selection_summary.json: {selection_summary_path}")

    rows = _read_csv(selected_metadata_path)
    if patch_index < 0 or patch_index >= len(rows):
        raise IndexError(f"Patch index {patch_index} outside selected_metadata.csv range.")
    row = rows[patch_index]
    summary = _read_json(selection_summary_path)
    filename = row.get("filename", "")
    patch_id = _first_non_empty(row.get("patch_id"), Path(filename).stem, f"patch_{patch_index:04d}")
    wsi_path = _first_non_empty(row.get("source_wsi_path"), summary.get("source_wsi_path"), summary.get("wsi_path"))
    if not wsi_path:
        raise ValueError("Missing WSI path in selected_metadata.csv/selection_summary.json.")
    x_level0 = _safe_int(row.get("x_level0"))
    y_level0 = _safe_int(row.get("y_level0"))
    patch_size = _safe_int(row.get("patch_size"), PATCH_SIZE)
    if x_level0 is None or y_level0 is None:
        raise ValueError(f"Missing x_level0/y_level0 for patch index {patch_index}.")
    if patch_size != PATCH_SIZE:
        raise ValueError(f"This probe expects 1024x1024 patches, got patch_size={patch_size}.")

    patch_png = selection_dir / "selected" / filename
    return {
        "patch_index": patch_index,
        "patch_id": patch_id,
        "filename": filename,
        "selected_patch_png": str(patch_png) if filename else None,
        "wsi_path": str(Path(wsi_path).expanduser().resolve()),
        "x_level0": x_level0,
        "y_level0": y_level0,
        "patch_size": patch_size,
        "mpp_x": _safe_float(_first_non_empty(row.get("mpp_x"), summary.get("mpp_x"))),
        "mpp_y": _safe_float(_first_non_empty(row.get("mpp_y"), summary.get("mpp_y"))),
        "objective_power": _first_non_empty(row.get("objective_power"), summary.get("objective_power")),
        "selection_method": _first_non_empty(row.get("selection_method"), summary.get("selector")),
        "selected_metadata_path": str(selected_metadata_path),
        "selection_summary_path": str(selection_summary_path),
    }


def _import_openslide() -> object:
    try:
        return importlib.import_module("openslide")
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic
        raise RuntimeError("Missing dependency: openslide. Activate inf402-lumina-seg.") from exc


def _read_wsi_bounds(
    slide: object,
    *,
    bounds: tuple[int, int, int, int],
    output_size: int = PATCH_INPUT_SHAPE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one RGB window from WSI level 0 with white padding outside slide bounds."""
    x0, y0, x1, y1 = [int(value) for value in bounds]
    slide_width, slide_height = slide.dimensions
    read_x0 = max(0, x0)
    read_y0 = max(0, y0)
    read_x1 = min(slide_width, x1)
    read_y1 = min(slide_height, y1)
    read_width = max(0, read_x1 - read_x0)
    read_height = max(0, read_y1 - read_y0)
    padding_left = max(0, -x0)
    padding_top = max(0, -y0)
    padding_right = max(0, x1 - slide_width)
    padding_bottom = max(0, y1 - slide_height)

    canvas = Image.new("RGB", (output_size, output_size), PADDING_RGB)
    if read_width > 0 and read_height > 0:
        region = slide.read_region((read_x0, read_y0), 0, (read_width, read_height)).convert("RGB")
        canvas.paste(region, (padding_left, padding_top))
    padding = {
        "padding_used": any(value > 0 for value in (padding_left, padding_right, padding_top, padding_bottom)),
        "padding_left": int(padding_left),
        "padding_right": int(padding_right),
        "padding_top": int(padding_top),
        "padding_bottom": int(padding_bottom),
        "padding_mode": "white",
        "read_bounds_level0": [int(read_x0), int(read_y0), int(read_x1), int(read_y1)],
    }
    return np.asarray(canvas, dtype=np.uint8), padding


def _read_context_preview(
    slide: object,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry = compute_context_geometry(
        x_level0=int(metadata["x_level0"]),
        y_level0=int(metadata["y_level0"]),
        patch_size=PATCH_SIZE,
        slide_width=slide.dimensions[0],
        slide_height=slide.dimensions[1],
    )
    bounds = (
        int(geometry["context_x0_requested"]),
        int(geometry["context_y0_requested"]),
        int(geometry["context_x0_requested"]) + int(geometry["context_width"]),
        int(geometry["context_y0_requested"]) + int(geometry["context_height"]),
    )
    context, padding = _read_wsi_bounds(slide, bounds=bounds, output_size=int(geometry["context_width"]))
    return context, {"geometry": geometry, "padding": padding}


def _window_record(
    *,
    strategy: str,
    window_id: str,
    target_x0: int,
    target_y0: int,
    input_bounds_target: tuple[int, int, int, int],
    output_bounds_target: tuple[int, int, int, int],
) -> dict[str, Any]:
    input_x0, input_y0, input_x1, input_y1 = input_bounds_target
    out_x0, out_y0, out_x1, out_y1 = output_bounds_target
    return {
        "strategy": strategy,
        "window_id": window_id,
        "input_bounds_target": [input_x0, input_y0, input_x1, input_y1],
        "input_bounds_wsi": [
            target_x0 + input_x0,
            target_y0 + input_y0,
            target_x0 + input_x1,
            target_y0 + input_y1,
        ],
        "output_bounds_target": [out_x0, out_y0, out_x1, out_y1],
        "output_bounds_wsi": [
            target_x0 + out_x0,
            target_y0 + out_y0,
            target_x0 + out_x1,
            target_y0 + out_y1,
        ],
        "clipped_output_bounds_target": [
            max(0, out_x0),
            max(0, out_y0),
            min(PATCH_SIZE, out_x1),
            min(PATCH_SIZE, out_y1),
        ],
    }


def no_overlap_window_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return four non-overlapping output-window records."""
    target_x0 = int(metadata["x_level0"])
    target_y0 = int(metadata["y_level0"])
    offsets = {
        "window_00": (0, 0),
        "window_01": (PATCH_OUTPUT_SHAPE, 0),
        "window_10": (0, PATCH_OUTPUT_SHAPE),
        "window_11": (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
    }
    records = []
    for window_id in WINDOW_IDS:
        out_x0, out_y0 = offsets[window_id]
        out_x1 = out_x0 + PATCH_OUTPUT_SHAPE
        out_y1 = out_y0 + PATCH_OUTPUT_SHAPE
        records.append(
            _window_record(
                strategy="no_overlap",
                window_id=window_id,
                target_x0=target_x0,
                target_y0=target_y0,
                input_bounds_target=(
                    out_x0 - CENTER_MARGIN,
                    out_y0 - CENTER_MARGIN,
                    out_x0 - CENTER_MARGIN + PATCH_INPUT_SHAPE,
                    out_y0 - CENTER_MARGIN + PATCH_INPUT_SHAPE,
                ),
                output_bounds_target=(out_x0, out_y0, out_x1, out_y1),
            )
        )
    return records


def overlap_aware_window_records(metadata: dict[str, Any], overlap_stride: int) -> list[dict[str, Any]]:
    """Use TIAToolbox coordinate logic to define overlap-aware windows."""
    ensure_runtime_environment()
    patchextraction = importlib.import_module("tiatoolbox.tools.patchextraction")
    PatchExtractor = patchextraction.PatchExtractor
    inputs, outputs = PatchExtractor.get_coordinates(
        image_shape=(PATCH_SIZE, PATCH_SIZE),
        patch_input_shape=(PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE),
        stride_shape=(overlap_stride, overlap_stride),
        patch_output_shape=(PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
    )
    target_x0 = int(metadata["x_level0"])
    target_y0 = int(metadata["y_level0"])
    records = []
    for idx, (input_bounds, output_bounds) in enumerate(zip(inputs, outputs, strict=True)):
        out_x0, out_y0, out_x1, out_y1 = [int(value) for value in output_bounds]
        if out_x1 <= 0 or out_y1 <= 0 or out_x0 >= PATCH_SIZE or out_y0 >= PATCH_SIZE:
            continue
        records.append(
            _window_record(
                strategy="overlap_aware",
                window_id=f"overlap_{idx:02d}",
                target_x0=target_x0,
                target_y0=target_y0,
                input_bounds_target=tuple(int(value) for value in input_bounds),
                output_bounds_target=(out_x0, out_y0, out_x1, out_y1),
            )
        )
    return records


def _load_windows_from_wsi(
    slide: object,
    records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    windows: list[np.ndarray] = []
    enriched: list[dict[str, Any]] = []
    for record in records:
        window_rgb, padding = _read_wsi_bounds(
            slide,
            bounds=tuple(int(value) for value in record["input_bounds_wsi"]),
            output_size=PATCH_INPUT_SHAPE,
        )
        item = dict(record)
        item["padding"] = padding
        windows.append(window_rgb)
        enriched.append(item)
    return windows, enriched


def build_segmentor_once(model_name: str, requested_device: str) -> tuple[object, str, dict[str, Any]]:
    """Create a single TIAToolbox SemanticSegmentor instance for all windows."""
    ensure_runtime_environment()
    torch_module = importlib.import_module("torch")
    tiatoolbox_module = importlib.import_module("tiatoolbox")
    resolved_device = resolve_torch_device(torch_module, requested_device)
    segmentor = _build_segmentor(model_name=model_name, resolved_device=resolved_device)
    metadata = {
        "torch_version": str(getattr(torch_module, "__version__", "version unavailable")),
        "tiatoolbox_version": str(getattr(tiatoolbox_module, "__version__", "version unavailable")),
        "model_class": f"{segmentor.model.__class__.__module__}.{segmentor.model.__class__.__name__}",
        "ioconfig_class": (
            f"{segmentor.ioconfig.__class__.__module__}.{segmentor.ioconfig.__class__.__name__}"
            if getattr(segmentor, "ioconfig", None) is not None
            else None
        ),
    }
    return segmentor, resolved_device, metadata


def _infer_single_window(
    segmentor: object,
    window_rgb: np.ndarray,
    *,
    model_name: str,
    resolved_device: str,
) -> dict[str, Any]:
    patch_batch = np.expand_dims(np.asarray(window_rgb, dtype=np.uint8), axis=0)
    start = time.perf_counter()
    run_output = segmentor.run(
        patch_batch,
        patch_mode=True,
        output_type="dict",
        return_probabilities=True,
        device=resolved_device,
        verbose=True,
    )
    runtime = time.perf_counter() - start
    label_mask, prediction_source = _prediction_from_run_output(run_output)
    probabilities, probability_source, probability_reason = _probability_from_run_output(run_output)
    if probabilities is None:
        raise RuntimeError(
            "Probabilities were not available for overlap-aware blending. "
            f"Reason: {probability_reason}"
        )
    probability_array, probability_shape_reason = _probability_array_for_label_mask(
        probabilities=probabilities,
        label_mask=label_mask,
    )
    if probability_array is None:
        raise RuntimeError(
            "Probability shape is incompatible with label mask. "
            f"Reason: {probability_shape_reason}"
        )
    if probability_array.shape[:2] != (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE):
        raise RuntimeError(
            f"Expected probability output 512x512, got {probability_array.shape[:2]}."
        )
    return {
        "label": np.asarray(label_mask, dtype=np.int32),
        "probabilities": probability_array.astype(np.float32, copy=False),
        "runtime_seconds": runtime,
        "prediction_source": prediction_source,
        "probability_source": probability_source,
        "model_name": model_name,
    }


def infer_patch_windows(
    segmentor: object,
    windows_rgb: list[np.ndarray],
    *,
    model_name: str,
    resolved_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Infer all windows while reusing one SemanticSegmentor instance."""
    results = []
    start = time.perf_counter()
    for window_rgb in windows_rgb:
        results.append(
            _infer_single_window(
                segmentor=segmentor,
                window_rgb=window_rgb,
                model_name=model_name,
                resolved_device=resolved_device,
            )
        )
    runtime = time.perf_counter() - start
    per_window = [float(item["runtime_seconds"]) for item in results]
    return results, {
        "runtime_seconds": runtime,
        "mean_runtime_per_window": float(np.mean(per_window)) if per_window else None,
        "per_window_runtime_seconds": per_window,
    }


def _feather_weights(shape: tuple[int, int], epsilon: float = 0.05) -> np.ndarray:
    y_size, x_size = shape
    y = np.hanning(y_size)
    x = np.hanning(x_size)
    weights = np.outer(y, x).astype(np.float32)
    if float(weights.max()) > 0:
        weights /= float(weights.max())
    return np.maximum(weights, epsilon).astype(np.float32)


def _weights_for_output(shape: tuple[int, int], blend_mode: str) -> np.ndarray:
    if blend_mode == "uniform":
        return np.ones(shape, dtype=np.float32)
    if blend_mode == "feathered":
        return _feather_weights(shape)
    raise ValueError(f"Unsupported blend_mode: {blend_mode}")


def _place_probabilities(
    *,
    records: list[dict[str, Any]],
    inference_results: list[dict[str, Any]],
    blend_mode: str,
) -> dict[str, Any]:
    if len(records) != len(inference_results):
        raise ValueError("records and inference_results length mismatch.")
    num_classes = int(inference_results[0]["probabilities"].shape[-1])
    prob_sum = np.zeros((PATCH_SIZE, PATCH_SIZE, num_classes), dtype=np.float32)
    weight_sum = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    coverage_count = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)

    for record, result in zip(records, inference_results, strict=True):
        probs = result["probabilities"]
        out_x0, out_y0, out_x1, out_y1 = [int(value) for value in record["output_bounds_target"]]
        clip_x0, clip_y0, clip_x1, clip_y1 = [
            int(value) for value in record["clipped_output_bounds_target"]
        ]
        if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
            continue
        src_x0 = clip_x0 - out_x0
        src_y0 = clip_y0 - out_y0
        src_x1 = src_x0 + (clip_x1 - clip_x0)
        src_y1 = src_y0 + (clip_y1 - clip_y0)
        cropped_probs = probs[src_y0:src_y1, src_x0:src_x1, :]
        weights = _weights_for_output(cropped_probs.shape[:2], blend_mode=blend_mode)
        prob_sum[clip_y0:clip_y1, clip_x0:clip_x1, :] += cropped_probs * weights[..., None]
        weight_sum[clip_y0:clip_y1, clip_x0:clip_x1] += weights
        coverage_count[clip_y0:clip_y1, clip_x0:clip_x1] += 1

    if np.any(weight_sum <= 0):
        zero_pixels = int(np.sum(weight_sum <= 0))
        raise RuntimeError(f"Overlap reconstruction left {zero_pixels} target pixels without coverage.")
    blended_probabilities = prob_sum / weight_sum[..., None]
    labels = np.argmax(blended_probabilities, axis=-1).astype(np.int32)
    return {
        "labels": labels,
        "probabilities": blended_probabilities,
        "coverage_count": coverage_count,
        "weight_sum": weight_sum,
    }


def class_distribution(labels: np.ndarray) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int32)
    unique, counts = np.unique(label_array, return_counts=True)
    total = int(label_array.size)
    class_pixel_counts = {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)}
    class_pixel_ratios = {key: value / total for key, value in class_pixel_counts.items()}
    return {
        "class_pixel_counts": class_pixel_counts,
        "class_pixel_ratios": class_pixel_ratios,
        "num_classes_present": int(len(unique)),
    }


def probability_summary(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float32)
    eps = 1e-8
    max_prob = np.max(probs, axis=-1)
    entropy = -np.sum(probs * np.log(np.maximum(probs, eps)), axis=-1) / np.log(probs.shape[-1])
    mean_probability_by_predicted_class = {}
    for class_id in sorted(int(value) for value in np.unique(labels)):
        mask = labels == class_id
        mean_probability_by_predicted_class[str(class_id)] = float(np.mean(max_prob[mask]))
    return {
        "available": True,
        "probability_shape": list(probs.shape),
        "mean_max_probability": float(np.mean(max_prob)),
        "median_max_probability": float(np.median(max_prob)),
        "min_max_probability": float(np.min(max_prob)),
        "max_max_probability": float(np.max(max_prob)),
        "mean_entropy": float(np.mean(entropy)),
        "median_entropy": float(np.median(entropy)),
        "mean_probability_by_predicted_class": mean_probability_by_predicted_class,
    }


def coverage_summary(coverage_count: np.ndarray, weight_sum: np.ndarray) -> dict[str, Any]:
    total_pixels = int(coverage_count.size)
    return {
        "min_coverage_count": int(np.min(coverage_count)),
        "max_coverage_count": int(np.max(coverage_count)),
        "mean_coverage_count": float(np.mean(coverage_count)),
        "pixels_without_coverage": int(np.sum(coverage_count == 0)),
        "pixels_with_multiple_predictions": int(np.sum(coverage_count > 1)),
        "pct_pixels_with_multiple_predictions": float(np.sum(coverage_count > 1) / total_pixels),
        "min_accumulated_weight": float(np.min(weight_sum)),
        "max_accumulated_weight": float(np.max(weight_sum)),
        "mean_accumulated_weight": float(np.mean(weight_sum)),
    }


def _seam_positions_from_records(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    x_positions: set[int] = set()
    y_positions: set[int] = set()
    for record in records:
        x0, y0, x1, y1 = [int(value) for value in record["clipped_output_bounds_target"]]
        for x in (x0, x1):
            if 0 < x < PATCH_SIZE:
                x_positions.add(x)
        for y in (y0, y1):
            if 0 < y < PATCH_SIZE:
                y_positions.add(y)
    return {"x": sorted(x_positions), "y": sorted(y_positions)}


def seam_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seam_positions_x: list[int],
    seam_positions_y: list[int],
    band: int,
) -> dict[str, Any]:
    label_values = []
    prob_values = []
    max_prob_values = []

    for seam_x in seam_positions_x:
        if seam_x - band < 0 or seam_x + band > labels.shape[1]:
            continue
        left_labels = labels[:, seam_x - band:seam_x]
        right_labels = labels[:, seam_x:seam_x + band]
        left_probs = probabilities[:, seam_x - band:seam_x, :]
        right_probs = probabilities[:, seam_x:seam_x + band, :]
        label_values.append(float(np.mean(left_labels != right_labels)))
        prob_values.append(float(np.mean(np.abs(left_probs - right_probs))))
        max_prob_values.append(
            float(np.mean(np.abs(np.max(left_probs, axis=-1) - np.max(right_probs, axis=-1))))
        )

    for seam_y in seam_positions_y:
        if seam_y - band < 0 or seam_y + band > labels.shape[0]:
            continue
        top_labels = labels[seam_y - band:seam_y, :]
        bottom_labels = labels[seam_y:seam_y + band, :]
        top_probs = probabilities[seam_y - band:seam_y, :, :]
        bottom_probs = probabilities[seam_y:seam_y + band, :, :]
        label_values.append(float(np.mean(top_labels != bottom_labels)))
        prob_values.append(float(np.mean(np.abs(top_probs - bottom_probs))))
        max_prob_values.append(
            float(np.mean(np.abs(np.max(top_probs, axis=-1) - np.max(bottom_probs, axis=-1))))
        )

    return {
        "seam_positions_x": seam_positions_x,
        "seam_positions_y": seam_positions_y,
        "band_pixels_each_side": band,
        "seam_label_discontinuity": float(np.mean(label_values)) if label_values else None,
        "seam_probability_discontinuity": float(np.mean(prob_values)) if prob_values else None,
        "seam_max_probability_discontinuity": float(np.mean(max_prob_values)) if max_prob_values else None,
        "num_seams_evaluated": len(label_values),
    }


def _iou_by_class(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    classes = sorted({int(value) for value in np.unique(a)} | {int(value) for value in np.unique(b)})
    output = {}
    for class_id in classes:
        a_mask = a == class_id
        b_mask = b == class_id
        union = np.logical_or(a_mask, b_mask).sum()
        output[str(class_id)] = None if union == 0 else float(np.logical_and(a_mask, b_mask).sum() / union)
    return output


def _macro_iou(ious: dict[str, float | None]) -> float | None:
    values = [value for value in ious.values() if value is not None]
    return float(np.mean(values)) if values else None


def compare_label_outputs(no_overlap: dict[str, Any], overlap: dict[str, Any]) -> dict[str, Any]:
    a = no_overlap["labels"]
    b = overlap["labels"]
    pixel_agreement = float(np.mean(a == b))
    ious = _iou_by_class(a, b)
    a_ratios = class_distribution(a)["class_pixel_ratios"]
    b_ratios = class_distribution(b)["class_pixel_ratios"]
    all_classes = sorted(set(a_ratios) | set(b_ratios), key=int)
    ratio_diff = {
        class_id: abs(float(a_ratios.get(class_id, 0.0)) - float(b_ratios.get(class_id, 0.0)))
        for class_id in all_classes
    }
    return {
        "agreement_metric_warning": AGREEMENT_METRIC_WARNING,
        "pixel_agreement": pixel_agreement,
        "disagreement_pixel_ratio": float(1.0 - pixel_agreement),
        "iou_by_class_between_strategies": ious,
        "macro_iou_between_strategies": _macro_iou(ious),
        "absolute_class_ratio_difference": ratio_diff,
        "mean_absolute_class_ratio_difference": float(np.mean(list(ratio_diff.values()))) if ratio_diff else 0.0,
    }


def _save_grayscale(array: np.ndarray, path: Path) -> Path:
    arr = np.asarray(array, dtype=np.float32)
    if float(arr.max()) > float(arr.min()):
        scaled = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
    else:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    Image.fromarray(scaled, mode="L").save(path)
    return path


def _save_strategy_outputs(
    strategy_dir: Path,
    *,
    target_rgb: np.ndarray,
    strategy_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    coverage_count: np.ndarray,
    weight_sum: np.ndarray,
    records: list[dict[str, Any]],
    runtime: dict[str, Any],
    blend_mode: str,
    resolved_device: str,
    model_name: str,
    seam_band: int,
    overlay_alpha: float,
) -> dict[str, Any]:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    labels_path = strategy_dir / "prediction_labels.npy"
    mask_path = strategy_dir / "prediction_mask.png"
    overlay_path = strategy_dir / "prediction_overlay.png"
    probability_summary_path = strategy_dir / "probability_summary.json"
    manifest_path = strategy_dir / "strategy_manifest.json"
    np.save(labels_path, labels)
    Image.fromarray(colorize_label_mask(labels)).save(mask_path)
    overlay = overlay_label_mask(target_rgb, labels, alpha=overlay_alpha)
    Image.fromarray(overlay).save(overlay_path)

    coverage_path = strategy_dir / "coverage_count_map.png"
    weights_path = strategy_dir / "accumulated_weights.png"
    _save_grayscale(coverage_count, coverage_path)
    _save_grayscale(weight_sum, weights_path)

    if strategy_name == "no_overlap":
        seam_positions = {"x": [512], "y": [512]}
    else:
        seam_positions = _seam_positions_from_records(records)
        if 512 not in seam_positions["x"]:
            seam_positions["x"].append(512)
        if 512 not in seam_positions["y"]:
            seam_positions["y"].append(512)
        seam_positions["x"] = sorted(set(seam_positions["x"]))
        seam_positions["y"] = sorted(set(seam_positions["y"]))

    seams = seam_metrics(
        labels,
        probabilities,
        seam_positions_x=seam_positions["x"],
        seam_positions_y=seam_positions["y"],
        band=seam_band,
    )
    distribution = class_distribution(labels)
    prob_summary = probability_summary(probabilities, labels)
    coverage = coverage_summary(coverage_count, weight_sum)
    write_json(prob_summary, probability_summary_path)

    seam_locations_path = strategy_dir / "seam_locations.png"
    seam_image = Image.fromarray(target_rgb).convert("RGB")
    draw = ImageDraw.Draw(seam_image, "RGBA")
    for x in seam_positions["x"]:
        draw.line([(x, 0), (x, PATCH_SIZE - 1)], fill=(255, 0, 0, 180), width=3)
    for y in seam_positions["y"]:
        draw.line([(0, y), (PATCH_SIZE - 1, y)], fill=(255, 0, 0, 180), width=3)
    seam_image.save(seam_locations_path)

    manifest = {
        "strategy": strategy_name,
        "model_name": model_name,
        "resolved_device": resolved_device,
        "blend_mode": blend_mode,
        "num_windows": len(records),
        "patch_input_shape": [PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE],
        "patch_output_shape": [PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE],
        "target_bounds": [0, 0, PATCH_SIZE, PATCH_SIZE],
        "input_bounds": [record["input_bounds_wsi"] for record in records],
        "output_bounds": [record["output_bounds_target"] for record in records],
        "clipped_output_bounds": [record["clipped_output_bounds_target"] for record in records],
        "windows": records,
        "runtime": runtime,
        "class_distribution": distribution,
        "probability_summary": prob_summary,
        "seam_metrics": seams,
        "coverage": coverage,
        "outputs": {
            "prediction_labels_npy": str(labels_path),
            "prediction_mask_png": str(mask_path),
            "prediction_overlay_png": str(overlay_path),
            "probability_summary_json": str(probability_summary_path),
            "coverage_count_map_png": str(coverage_path),
            "accumulated_weights_png": str(weights_path),
            "seam_locations_png": str(seam_locations_path),
        },
        "clinical_warning": CLINICAL_WARNING,
    }
    write_json(manifest, manifest_path)
    manifest["outputs"]["strategy_manifest_json"] = str(manifest_path)
    return {
        "name": strategy_name,
        "labels": labels,
        "probabilities": probabilities,
        "coverage_count": coverage_count,
        "weight_sum": weight_sum,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "overlay_path": str(overlay_path),
    }


def _strategy_from_records(
    *,
    records: list[dict[str, Any]],
    inference_results: list[dict[str, Any]],
    blend_mode: str,
) -> dict[str, Any]:
    return _place_probabilities(
        records=records,
        inference_results=inference_results,
        blend_mode=blend_mode,
    )


def _make_side_by_side(
    *,
    target_rgb: np.ndarray,
    no_overlap_overlay: Path,
    overlap_overlay: Path,
    disagreement_mask: np.ndarray,
    output_path: Path,
) -> Path:
    panels = [
        Image.fromarray(target_rgb).convert("RGB"),
        Image.open(no_overlap_overlay).convert("RGB"),
        Image.open(overlap_overlay).convert("RGB"),
        Image.fromarray(disagreement_mask).convert("RGB"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _save_source_images(
    *,
    source_dir: Path,
    slide: object,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_dir.mkdir(parents=True, exist_ok=True)
    target_bounds = (
        int(metadata["x_level0"]),
        int(metadata["y_level0"]),
        int(metadata["x_level0"]) + PATCH_SIZE,
        int(metadata["y_level0"]) + PATCH_SIZE,
    )
    target_rgb, target_padding = _read_wsi_bounds(slide, bounds=target_bounds, output_size=PATCH_SIZE)
    context_rgb, context_info = _read_context_preview(slide, metadata)
    Image.fromarray(target_rgb).save(source_dir / "target_patch.png")
    Image.fromarray(context_rgb).save(source_dir / "context_preview.png")
    return target_rgb, {
        "target_patch_png": str(source_dir / "target_patch.png"),
        "context_preview_png": str(source_dir / "context_preview.png"),
        "target_padding": target_padding,
        "context_info": context_info,
    }


def compare_one_patch(
    *,
    metadata: dict[str, Any],
    patch_output_dir: Path,
    segmentor: object,
    resolved_device: str,
    model_metadata: dict[str, Any],
    config: StrategyComparisonConfig,
    blend_mode: str,
) -> dict[str, Any]:
    """Compare strategies for one selected patch."""
    patch_start = time.perf_counter()
    openslide_module = _import_openslide()
    slide = openslide_module.OpenSlide(str(metadata["wsi_path"]))
    try:
        source_dir = patch_output_dir / "source"
        target_rgb, source_outputs = _save_source_images(
            source_dir=source_dir,
            slide=slide,
            metadata=metadata,
        )

        strategy_outputs: dict[str, Any] = {}
        if config.run_no_overlap:
            records = no_overlap_window_records(metadata)
            windows, enriched_records = _load_windows_from_wsi(slide, records)
            inference_results, runtime = infer_patch_windows(
                segmentor,
                windows,
                model_name=config.model_name,
                resolved_device=resolved_device,
            )
            reconstructed = _strategy_from_records(
                records=enriched_records,
                inference_results=inference_results,
                blend_mode="uniform",
            )
            strategy_outputs["no_overlap"] = _save_strategy_outputs(
                patch_output_dir / "no_overlap",
                target_rgb=target_rgb,
                strategy_name="no_overlap",
                labels=reconstructed["labels"],
                probabilities=reconstructed["probabilities"],
                coverage_count=reconstructed["coverage_count"],
                weight_sum=reconstructed["weight_sum"],
                records=enriched_records,
                runtime=runtime,
                blend_mode="uniform",
                resolved_device=resolved_device,
                model_name=config.model_name,
                seam_band=config.seam_band,
                overlay_alpha=config.overlay_alpha,
            )

        if config.run_overlap_aware:
            records = overlap_aware_window_records(metadata, overlap_stride=config.overlap_stride)
            windows, enriched_records = _load_windows_from_wsi(slide, records)
            inference_results, runtime = infer_patch_windows(
                segmentor,
                windows,
                model_name=config.model_name,
                resolved_device=resolved_device,
            )
            reconstructed = _strategy_from_records(
                records=enriched_records,
                inference_results=inference_results,
                blend_mode=blend_mode,
            )
            strategy_name = f"overlap_{blend_mode}"
            strategy_outputs[strategy_name] = _save_strategy_outputs(
                patch_output_dir / strategy_name,
                target_rgb=target_rgb,
                strategy_name=strategy_name,
                labels=reconstructed["labels"],
                probabilities=reconstructed["probabilities"],
                coverage_count=reconstructed["coverage_count"],
                weight_sum=reconstructed["weight_sum"],
                records=enriched_records,
                runtime=runtime,
                blend_mode=blend_mode,
                resolved_device=resolved_device,
                model_name=config.model_name,
                seam_band=config.seam_band,
                overlay_alpha=config.overlay_alpha,
            )
    finally:
        slide.close()

    comparison: dict[str, Any] | None = None
    if "no_overlap" in strategy_outputs:
        overlap_keys = [key for key in strategy_outputs if key.startswith("overlap_")]
        if overlap_keys:
            overlap_key = overlap_keys[0]
            no_overlap = strategy_outputs["no_overlap"]
            overlap = strategy_outputs[overlap_key]
            comparison_dir = patch_output_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            metrics = compare_label_outputs(no_overlap, overlap)
            disagreement = (no_overlap["labels"] != overlap["labels"]).astype(np.uint8) * 255
            disagreement_path = comparison_dir / "disagreement_mask.png"
            Image.fromarray(disagreement, mode="L").save(disagreement_path)
            side_by_side_path = _make_side_by_side(
                target_rgb=target_rgb,
                no_overlap_overlay=Path(no_overlap["overlay_path"]),
                overlap_overlay=Path(overlap["overlay_path"]),
                disagreement_mask=np.stack([disagreement] * 3, axis=-1),
                output_path=comparison_dir / "side_by_side.png",
            )
            metrics["outputs"] = {
                "disagreement_mask_png": str(disagreement_path),
                "side_by_side_png": str(side_by_side_path),
            }
            metrics_path = write_json(metrics, comparison_dir / "comparison_metrics.json")
            comparison = {"metrics": metrics, "metrics_path": str(metrics_path)}

    patch_runtime = time.perf_counter() - patch_start
    patch_summary = {
        "status": "completed",
        "patch_index": metadata["patch_index"],
        "patch_id": metadata["patch_id"],
        "metadata": metadata,
        "source_outputs": source_outputs,
        "model_metadata": model_metadata,
        "resolved_device": resolved_device,
        "blend_mode": blend_mode,
        "runtime_seconds": patch_runtime,
        "strategies": {
            key: {
                "manifest_path": value["manifest_path"],
                "num_windows": value["manifest"]["num_windows"],
                "runtime": value["manifest"]["runtime"],
                "class_distribution": value["manifest"]["class_distribution"],
                "probability_summary": value["manifest"]["probability_summary"],
                "seam_metrics": value["manifest"]["seam_metrics"],
                "coverage": value["manifest"]["coverage"],
            }
            for key, value in strategy_outputs.items()
        },
        "comparison": comparison,
        "clinical_warning": CLINICAL_WARNING,
    }
    write_json(patch_summary, patch_output_dir / "patch_comparison_summary.json")
    return patch_summary


def _mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def aggregate_results(patch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate patch-level comparison summaries."""
    strategy_names = sorted(
        {
            strategy_name
            for patch in patch_summaries
            for strategy_name in patch.get("strategies", {})
        }
    )
    strategies: dict[str, Any] = {}
    for strategy_name in strategy_names:
        items = [
            patch["strategies"][strategy_name]
            for patch in patch_summaries
            if strategy_name in patch.get("strategies", {})
        ]
        class_ids = sorted(
            {
                class_id
                for item in items
                for class_id in item["class_distribution"]["class_pixel_ratios"]
            },
            key=int,
        )
        class_ratios: dict[str, list[float]] = {class_id: [] for class_id in class_ids}
        for item in items:
            ratios = item["class_distribution"]["class_pixel_ratios"]
            for class_id in class_ids:
                class_ratios[class_id].append(float(ratios.get(class_id, 0.0)))
        strategies[strategy_name] = {
            "num_patches": len(items),
            "mean_runtime_seconds": _mean([item["runtime"]["runtime_seconds"] for item in items]),
            "mean_num_windows": _mean([item["num_windows"] for item in items]),
            "mean_seam_label_discontinuity": _mean(
                [item["seam_metrics"]["seam_label_discontinuity"] for item in items]
            ),
            "mean_seam_probability_discontinuity": _mean(
                [item["seam_metrics"]["seam_probability_discontinuity"] for item in items]
            ),
            "mean_max_probability": _mean(
                [item["probability_summary"]["mean_max_probability"] for item in items]
            ),
            "mean_entropy": _mean([item["probability_summary"]["mean_entropy"] for item in items]),
            "mean_class_pixel_ratios": {
                class_id: float(np.mean(values)) for class_id, values in sorted(class_ratios.items(), key=lambda kv: int(kv[0]))
            },
            "mean_pct_pixels_with_multiple_predictions": _mean(
                [item["coverage"]["pct_pixels_with_multiple_predictions"] for item in items]
            ),
            "mean_pixels_without_coverage": _mean(
                [item["coverage"]["pixels_without_coverage"] for item in items]
            ),
        }

    comparisons = [
        patch["comparison"]["metrics"]
        for patch in patch_summaries
        if patch.get("comparison") is not None
    ]
    aggregate_comparison = {
        "num_patches": len(comparisons),
        "mean_pixel_agreement": _mean([item["pixel_agreement"] for item in comparisons]),
        "mean_disagreement_pixel_ratio": _mean(
            [item["disagreement_pixel_ratio"] for item in comparisons]
        ),
        "mean_macro_iou_between_strategies": _mean(
            [item["macro_iou_between_strategies"] for item in comparisons]
        ),
        "mean_absolute_class_ratio_difference": _mean(
            [item["mean_absolute_class_ratio_difference"] for item in comparisons]
        ),
    }

    recommendation, reasoning = recommend_strategy(strategies, aggregate_comparison)
    return {
        "status": "completed",
        "num_patches_processed": len(patch_summaries),
        "ground_truth_available": False,
        "accuracy_claim_allowed": False,
        "agreement_metric_warning": AGREEMENT_METRIC_WARNING,
        "strategies": strategies,
        "comparison": aggregate_comparison,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "clinical_warning": CLINICAL_WARNING,
    }


def recommend_strategy(
    strategies: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[str, list[str]]:
    """Make a bounded technical recommendation without accuracy claims."""
    reasoning: list[str] = []
    no_overlap = strategies.get("no_overlap")
    overlap_keys = [key for key in strategies if key.startswith("overlap_")]
    if no_overlap is None or not overlap_keys:
        return "inconclusive", ["Both no-overlap and overlap-aware outputs are required."]
    overlap = strategies[overlap_keys[0]]
    disagreement = comparison.get("mean_disagreement_pixel_ratio")
    no_seam = no_overlap.get("mean_seam_probability_discontinuity")
    overlap_seam = overlap.get("mean_seam_probability_discontinuity")
    no_runtime = no_overlap.get("mean_runtime_seconds")
    overlap_runtime = overlap.get("mean_runtime_seconds")
    overlap_uncovered = overlap.get("mean_pixels_without_coverage")

    if overlap_uncovered and overlap_uncovered > 0:
        return "prefer_no_overlap", ["Overlap-aware left pixels without coverage."]
    if disagreement is None:
        return "inconclusive", ["No agreement comparison was available."]

    if no_runtime and overlap_runtime:
        runtime_ratio = overlap_runtime / no_runtime
        reasoning.append(f"Runtime ratio overlap/no-overlap was {runtime_ratio:.2f}.")
    else:
        runtime_ratio = None

    if no_seam is not None and overlap_seam is not None:
        reasoning.append(
            "Mean seam probability discontinuity: "
            f"no-overlap={no_seam:.6f}, overlap={overlap_seam:.6f}."
        )
    reasoning.append(f"Mean disagreement ratio between strategies was {disagreement:.6f}.")

    if disagreement < 0.01 and (runtime_ratio is None or runtime_ratio > 1.5):
        reasoning.append(
            "Predicted labels were technically very similar while overlap-aware required more windows."
        )
        return "prefer_no_overlap", reasoning
    if (
        no_seam is not None
        and overlap_seam is not None
        and overlap_seam < no_seam * 0.9
        and (runtime_ratio is None or runtime_ratio < 3.5)
    ):
        reasoning.append("Overlap-aware reduced seam probability discontinuity with acceptable overhead.")
        return "prefer_overlap_aware", reasoning
    if disagreement < 0.03:
        reasoning.append("Differences were small under the current technical metrics.")
        return "technically_similar", reasoning
    return "inconclusive", reasoning


def write_metrics_csv(summary: dict[str, Any], path: Path) -> Path:
    """Write a compact strategy metrics CSV."""
    rows = []
    for strategy_name, item in summary.get("strategies", {}).items():
        rows.append(
            {
                "strategy": strategy_name,
                "num_patches": item.get("num_patches"),
                "mean_runtime_seconds": item.get("mean_runtime_seconds"),
                "mean_num_windows": item.get("mean_num_windows"),
                "mean_seam_label_discontinuity": item.get("mean_seam_label_discontinuity"),
                "mean_seam_probability_discontinuity": item.get("mean_seam_probability_discontinuity"),
                "mean_max_probability": item.get("mean_max_probability"),
                "mean_entropy": item.get("mean_entropy"),
                "mean_pct_pixels_with_multiple_predictions": item.get(
                    "mean_pct_pixels_with_multiple_predictions"
                ),
                "mean_pixels_without_coverage": item.get("mean_pixels_without_coverage"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "num_patches",
                "mean_runtime_seconds",
                "mean_num_windows",
                "mean_seam_label_discontinuity",
                "mean_seam_probability_discontinuity",
                "mean_max_probability",
                "mean_entropy",
                "mean_pct_pixels_with_multiple_predictions",
                "mean_pixels_without_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def compare_context_stitch_strategies(config: StrategyComparisonConfig) -> dict[str, Any]:
    """Run the isolated strategy comparison over up to three selected patches."""
    if len(config.patch_indices) > 3:
        raise ValueError("Initial comparison is capped at 3 patches.")
    if not config.run_no_overlap and not config.run_overlap_aware:
        raise ValueError("At least one strategy must be enabled.")
    if config.blend_mode not in {"uniform", "feathered", "both"}:
        raise ValueError("blend_mode must be one of: uniform, feathered, both.")

    ensure_runtime_environment()
    segmentor, resolved_device, model_metadata = build_segmentor_once(
        model_name=config.model_name,
        requested_device=config.device,
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_blend_modes = (
        ("uniform", "feathered") if config.blend_mode == "both" else (config.blend_mode,)
    )
    patch_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.perf_counter()
    for patch_index in config.patch_indices:
        try:
            metadata = load_selected_patch_metadata(config.selection_dir, patch_index)
            patch_dir = output_dir / str(metadata["patch_id"])
            for blend_mode in selected_blend_modes:
                patch_blend_dir = patch_dir if len(selected_blend_modes) == 1 else patch_dir / f"blend_{blend_mode}"
                patch_summaries.append(
                    compare_one_patch(
                        metadata=metadata,
                        patch_output_dir=patch_blend_dir,
                        segmentor=segmentor,
                        resolved_device=resolved_device,
                        model_metadata=model_metadata,
                        config=config,
                        blend_mode=blend_mode,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - keep batch probe going
            failures.append(
                {
                    "patch_index": patch_index,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    aggregate = aggregate_results(patch_summaries) if patch_summaries else {
        "status": "failed",
        "recommendation": "inconclusive",
        "reasoning": ["No patch completed successfully."],
        "ground_truth_available": False,
        "accuracy_claim_allowed": False,
        "clinical_warning": CLINICAL_WARNING,
    }
    aggregate.update(
        {
            "selection_dir": str(config.selection_dir),
            "output_dir": str(output_dir),
            "patch_indices": list(config.patch_indices),
            "model_name": config.model_name,
            "requested_device": config.device,
            "resolved_device": resolved_device,
            "overlap_stride": config.overlap_stride,
            "blend_mode": config.blend_mode,
            "runtime_seconds": time.perf_counter() - start,
            "num_failures": len(failures),
            "failures": failures,
            "axis_convention": AXIS_CONVENTION,
        }
    )
    summary_path = write_json(aggregate, output_dir / "strategy_comparison_summary.json")
    csv_path = write_metrics_csv(aggregate, output_dir / "strategy_comparison_metrics.csv")
    aggregate["outputs"] = {
        "strategy_comparison_summary_json": str(summary_path),
        "strategy_comparison_metrics_csv": str(csv_path),
    }
    write_json(aggregate, summary_path)
    return aggregate
