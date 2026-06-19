"""Stable context-stitch 2x2 inference for selected WSI patches."""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.inference.context_stitching import (
    AXIS_CONVENTION,
    CLINICAL_WARNING,
    WINDOW_IDS,
    compute_context_geometry,
    extract_windows_2x2,
)
from src.inference.input_validation import validate_patch_input
from src.models.tiatoolbox_bcss import resolve_torch_device
from src.visualization.segmentation_overlay import colorize_label_mask, overlay_label_mask


PATCH_INPUT_SHAPE = 1024
PATCH_OUTPUT_SHAPE = 512
PATCH_SIZE = 1024
CONTEXT_MARGIN = (PATCH_INPUT_SHAPE - PATCH_OUTPUT_SHAPE) // 2
CONTEXT_SIZE = PATCH_SIZE + 2 * CONTEXT_MARGIN
PADDING_RGB = (255, 255, 255)
CLASS_COUNT_SOURCE = "stitched_prediction_1024"
INFERENCE_STRATEGY = "context-stitch-2x2"


@dataclass(frozen=True)
class ContextStitchRuntime:
    """Reusable runtime objects for context-stitch batch inference."""

    segmentor: object
    resolved_device: str
    model_metadata: dict[str, Any]


class SlideHandleCache:
    """Small OpenSlide handle cache for consecutive patches from the same WSI."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._slide: object | None = None

    def get(self, wsi_path: Path) -> object:
        path = Path(wsi_path).expanduser().resolve()
        if self._slide is not None and self._path == path:
            return self._slide
        self.close()
        openslide_module = importlib.import_module("openslide")
        self._slide = openslide_module.OpenSlide(str(path))
        self._path = path
        return self._slide

    def close(self) -> None:
        if self._slide is not None:
            close = getattr(self._slide, "close", None)
            if callable(close):
                close()
        self._slide = None
        self._path = None

    def __enter__(self) -> "SlideHandleCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def ensure_context_stitch_environment() -> None:
    """Set local cache dirs used by TIAToolbox if the caller did not provide them."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")


def _json_number(value: np.generic | int | float) -> int | float:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_context_stitch_segmentor(
    *,
    model_name: str,
    requested_device: str,
) -> ContextStitchRuntime:
    """Build one TIAToolbox SemanticSegmentor instance for the whole batch."""
    ensure_context_stitch_environment()
    torch_module = importlib.import_module("torch")
    tiatoolbox_module = importlib.import_module("tiatoolbox")
    semantic_module = importlib.import_module("tiatoolbox.models.engine.semantic_segmentor")
    resolved_device = resolve_torch_device(torch_module, requested_device)
    segmentor_class = getattr(semantic_module, "SemanticSegmentor")
    segmentor = segmentor_class(
        model=model_name,
        batch_size=1,
        num_workers=0,
        device=resolved_device,
        verbose=True,
    )
    model_metadata = {
        "model_name": model_name,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "torch_version": str(getattr(torch_module, "__version__", "version unavailable")),
        "tiatoolbox_version": str(
            getattr(tiatoolbox_module, "__version__", "version unavailable")
        ),
        "model_class": f"{segmentor.model.__class__.__module__}.{segmentor.model.__class__.__name__}",
        "ioconfig_class": (
            f"{segmentor.ioconfig.__class__.__module__}.{segmentor.ioconfig.__class__.__name__}"
            if getattr(segmentor, "ioconfig", None) is not None
            else None
        ),
        "batching_decision": (
            "Sequential four-window inference using one SemanticSegmentor instance. "
            "This preserves the behavior validated in the context-stitch probe while "
            "avoiding model reloads per window."
        ),
    }
    return ContextStitchRuntime(
        segmentor=segmentor,
        resolved_device=resolved_device,
        model_metadata=model_metadata,
    )


def _safe_int(value: object, field_name: str) -> int:
    try:
        if value in ("", None):
            raise ValueError
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field_name}: {value!r}") from exc


def _safe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None


def _first_non_empty(*values: object) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""


def _normalize_label_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[-1] <= 32:
            array = np.argmax(array, axis=-1)
        elif array.shape[0] <= 32:
            array = np.argmax(array, axis=0)
        else:
            raise ValueError(f"Cannot infer class axis for prediction shape {array.shape}.")
    if array.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got {array.shape}.")
    return array.astype(np.int32, copy=False)


def _to_numpy_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "compute"):
        return np.asarray(value.compute())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _value_from_nested_dict(payload: dict[Any, Any], keys: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key in payload:
            return _to_numpy_array(payload[key]), key
    if len(payload) == 1:
        only_key, only_value = next(iter(payload.items()))
        if isinstance(only_value, dict):
            value, source = _value_from_nested_dict(only_value, keys)
            if value is not None:
                return value, f"{only_key}.{source}"
    for key, value in payload.items():
        if isinstance(value, dict):
            nested, source = _value_from_nested_dict(value, keys)
            if nested is not None:
                return nested, f"{key}.{source}"
    return None, None


def _extract_prediction_and_probability(run_output: object) -> tuple[np.ndarray, np.ndarray, str, str]:
    if not isinstance(run_output, dict):
        raise RuntimeError(
            f"Context-stitch expects TIAToolbox dict output, got {type(run_output).__name__}."
        )
    prediction_value, prediction_source = _value_from_nested_dict(
        run_output,
        ("predictions", "prediction", "labels", "label", "mask", "masks"),
    )
    probability_value, probability_source = _value_from_nested_dict(
        run_output,
        ("probabilities", "probability", "probs"),
    )
    if prediction_value is None:
        if probability_value is None:
            raise RuntimeError(f"No predictions or probabilities found in output keys: {list(run_output)}")
        prediction_value = probability_value
        prediction_source = f"{probability_source}_argmax"
    if probability_value is None:
        raise RuntimeError(
            "Probabilities were not available for context-stitch reconstruction. "
            "The stitched mask is intentionally not built by interpolating labels."
        )
    labels = _normalize_label_mask(prediction_value)
    probabilities = np.asarray(probability_value)
    probabilities = np.squeeze(probabilities)
    if probabilities.ndim != 3:
        raise RuntimeError(f"Probability array must be 3D after squeeze, got {probabilities.shape}.")
    if probabilities.shape[:2] == labels.shape:
        probability_array = probabilities.astype(np.float32, copy=False)
    elif probabilities.shape[1:] == labels.shape:
        probability_array = np.moveaxis(probabilities, 0, -1).astype(np.float32, copy=False)
    else:
        raise RuntimeError(
            "Probability spatial shape does not match label mask: "
            f"probabilities={probabilities.shape}, labels={labels.shape}."
        )
    return labels, probability_array, str(prediction_source), str(probability_source)


def _read_context_from_wsi(
    slide: object,
    geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    context = Image.new("RGB", (CONTEXT_SIZE, CONTEXT_SIZE), PADDING_RGB)
    read_width = int(geometry["read_width"])
    read_height = int(geometry["read_height"])
    if read_width <= 0 or read_height <= 0:
        raise RuntimeError("Requested context has no valid overlap with the WSI.")
    read_x0 = int(geometry["read_x0"])
    read_y0 = int(geometry["read_y0"])
    region = slide.read_region((read_x0, read_y0), 0, (read_width, read_height)).convert("RGB")
    padding = dict(geometry["padding"])
    context.paste(region, (int(padding["padding_left"]), int(padding["padding_top"])))
    return np.asarray(context, dtype=np.uint8), padding


def _read_target_patch_from_wsi(slide: object, x_level0: int, y_level0: int) -> np.ndarray:
    target = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), PADDING_RGB)
    slide_width, slide_height = slide.dimensions
    read_x0 = max(0, x_level0)
    read_y0 = max(0, y_level0)
    read_x1 = min(slide_width, x_level0 + PATCH_SIZE)
    read_y1 = min(slide_height, y_level0 + PATCH_SIZE)
    read_width = max(0, read_x1 - read_x0)
    read_height = max(0, read_y1 - read_y0)
    if read_width <= 0 or read_height <= 0:
        raise RuntimeError("Target patch has no valid overlap with the WSI.")
    region = slide.read_region((read_x0, read_y0), 0, (read_width, read_height)).convert("RGB")
    target.paste(region, (max(0, -x_level0), max(0, -y_level0)))
    return np.asarray(target, dtype=np.uint8)


def _validate_metadata(selection_metadata: dict[str, object]) -> dict[str, Any]:
    wsi_path_text = _first_non_empty(
        selection_metadata.get("source_wsi_path"),
        selection_metadata.get("wsi_path"),
    )
    if not wsi_path_text:
        raise ValueError("Missing source_wsi_path/wsi_path for context-stitch inference.")
    wsi_path = Path(wsi_path_text).expanduser().resolve()
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")
    patch_size = _safe_int(selection_metadata.get("patch_size"), "patch_size")
    if patch_size != PATCH_SIZE:
        raise ValueError(f"context-stitch-2x2 requires patch_size=1024, got {patch_size}.")
    x_level0 = _safe_int(selection_metadata.get("x_level0"), "x_level0")
    y_level0 = _safe_int(selection_metadata.get("y_level0"), "y_level0")
    return {
        "wsi_path": wsi_path,
        "x_level0": x_level0,
        "y_level0": y_level0,
        "patch_size": patch_size,
        "mpp_x": _safe_float(selection_metadata.get("mpp_x")),
        "mpp_y": _safe_float(selection_metadata.get("mpp_y")),
    }


def _infer_context_windows(
    runtime: ContextStitchRuntime,
    windows: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    runtimes: list[float] = []
    for window_id in WINDOW_IDS:
        window = np.asarray(windows[window_id], dtype=np.uint8)
        if window.shape != (PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE, 3):
            raise RuntimeError(f"{window_id} input expected 1024x1024x3, got {window.shape}.")
        patch_batch = np.expand_dims(window.copy(), axis=0)
        start = time.perf_counter()
        run_output = runtime.segmentor.run(
            patch_batch,
            patch_mode=True,
            output_type="dict",
            return_probabilities=True,
            device=runtime.resolved_device,
            verbose=True,
        )
        window_runtime = time.perf_counter() - start
        labels, probabilities, prediction_source, probability_source = (
            _extract_prediction_and_probability(run_output)
        )
        if labels.shape != (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE):
            raise RuntimeError(f"{window_id} output expected 512x512, got {labels.shape}.")
        if probabilities.shape[:2] != (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE):
            raise RuntimeError(
                f"{window_id} probabilities expected 512x512xC, got {probabilities.shape}."
            )
        outputs[window_id] = {
            "labels": labels,
            "probabilities": probabilities,
            "runtime_seconds": window_runtime,
            "prediction_source": prediction_source,
            "probability_source": probability_source,
        }
        runtimes.append(window_runtime)
    return outputs, {
        "per_window_runtime_seconds": runtimes,
        "runtime_seconds": float(sum(runtimes)),
        "mean_runtime_per_window": float(np.mean(runtimes)) if runtimes else None,
    }


def _stitch_window_probabilities(
    window_outputs: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [window_id for window_id in WINDOW_IDS if window_id not in window_outputs]
    if missing:
        raise RuntimeError(f"Missing context-stitch window outputs: {missing}")
    num_classes = int(window_outputs["window_00"]["probabilities"].shape[-1])
    stitched_probabilities = np.zeros((PATCH_SIZE, PATCH_SIZE, num_classes), dtype=np.float32)
    coverage = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    placements = {
        "window_00": (0, 0),
        "window_01": (PATCH_OUTPUT_SHAPE, 0),
        "window_10": (0, PATCH_OUTPUT_SHAPE),
        "window_11": (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE),
    }
    for window_id in WINDOW_IDS:
        target_x0, target_y0 = placements[window_id]
        target_x1 = target_x0 + PATCH_OUTPUT_SHAPE
        target_y1 = target_y0 + PATCH_OUTPUT_SHAPE
        probabilities = window_outputs[window_id]["probabilities"]
        stitched_probabilities[target_y0:target_y1, target_x0:target_x1, :] = probabilities
        coverage[target_y0:target_y1, target_x0:target_x1] += 1
    if np.any(coverage == 0):
        raise RuntimeError("Stitched context output has pixels without probability coverage.")
    stitched_labels = np.argmax(stitched_probabilities, axis=-1).astype(np.int32)
    if stitched_labels.shape != (PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(f"Stitched labels expected 1024x1024, got {stitched_labels.shape}.")
    return stitched_labels, stitched_probabilities, coverage


def _class_distribution(labels: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    unique, counts = np.unique(labels, return_counts=True)
    total = int(labels.size)
    class_pixel_counts = {
        str(int(class_id)): int(count)
        for class_id, count in zip(unique, counts, strict=True)
    }
    class_pixel_ratios = {
        class_id: count / total
        for class_id, count in class_pixel_counts.items()
    }
    return class_pixel_counts, class_pixel_ratios


def _probability_summary(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    max_probability = np.max(probabilities, axis=-1)
    eps = 1e-8
    entropy = (
        -np.sum(probabilities * np.log(np.maximum(probabilities, eps)), axis=-1)
        / np.log(probabilities.shape[-1])
    )
    mean_probability_by_predicted_class: dict[str, float] = {}
    for class_id in sorted(int(value) for value in np.unique(labels)):
        class_mask = labels == class_id
        mean_probability_by_predicted_class[str(class_id)] = float(
            np.mean(max_probability[class_mask])
        )
    return {
        "available": True,
        "source": "stitched_window_probabilities",
        "probability_shape": list(probabilities.shape),
        "mean_max_probability": float(np.mean(max_probability)),
        "median_max_probability": float(np.median(max_probability)),
        "min_max_probability": float(np.min(max_probability)),
        "max_max_probability": float(np.max(max_probability)),
        "mean_entropy": float(np.mean(entropy)),
        "median_entropy": float(np.median(entropy)),
        "mean_probability_by_predicted_class": mean_probability_by_predicted_class,
    }


def _write_context_artifacts(
    output_dir: Path,
    context_rgb: np.ndarray,
    windows: dict[str, np.ndarray],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    context_path = output_dir / "context_preview.png"
    Image.fromarray(context_rgb).save(context_path)
    artifacts["context_preview"] = str(context_path)
    for window_id in WINDOW_IDS:
        window_path = output_dir / f"{window_id}_input.png"
        Image.fromarray(windows[window_id]).save(window_path)
        artifacts[f"{window_id}_input"] = str(window_path)
    return artifacts


def run_context_stitch_patch(
    *,
    patch_id: str,
    selected_patch_path: Path,
    patch_output_dir: Path,
    selection_metadata: dict[str, object],
    runtime: ContextStitchRuntime,
    slide_cache: SlideHandleCache,
    overlay_alpha: float,
    strict_input_validation: bool,
    save_probabilities: bool,
    save_context_artifacts: bool,
) -> tuple[dict[str, Any], Path]:
    """Run context-stitch-2x2 inference for one selected patch."""
    start = time.perf_counter()
    patch_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = patch_output_dir / "inference_summary.json"
    summary: dict[str, Any] = {
        "status": "failed",
        "inference_strategy": INFERENCE_STRATEGY,
        "patch_id": patch_id,
        "model_name": runtime.model_metadata["model_name"],
        "requested_device": runtime.model_metadata.get("requested_device", ""),
        "resolved_device": runtime.resolved_device,
        "patch_input_shape": [PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE],
        "window_prediction_shape": [PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE],
        "stitched_prediction_shape": None,
        "num_inference_windows": 4,
        "context_margin": CONTEXT_MARGIN,
        "context_shape": [CONTEXT_SIZE, CONTEXT_SIZE, 3],
        "class_count_source": CLASS_COUNT_SOURCE,
        "class_pixel_counts": {},
        "class_pixel_ratios": {},
        "probability_summary": {},
        "runtime_seconds": None,
        "mean_runtime_per_window": None,
        "padding": {},
        "outputs": {},
        "selection_metadata": selection_metadata,
        "input_validation": {},
        "model_metadata": runtime.model_metadata,
        "warnings": [],
        "error": None,
        "clinical_warning": CLINICAL_WARNING,
    }

    try:
        selected_rgb, input_validation = validate_patch_input(
            Path(selected_patch_path),
            selection_metadata=selection_metadata,
        )
        summary["input_validation"] = input_validation
        summary["warnings"].extend(str(warning) for warning in input_validation.get("warnings", []))
        if strict_input_validation and input_validation.get("strict_validation_failed"):
            raise ValueError("Strict input validation failed for selected patch PNG.")

        metadata = _validate_metadata(selection_metadata)
        slide = slide_cache.get(metadata["wsi_path"])
        slide_width, slide_height = slide.dimensions
        geometry = compute_context_geometry(
            x_level0=int(metadata["x_level0"]),
            y_level0=int(metadata["y_level0"]),
            patch_size=PATCH_SIZE,
            patch_input_shape=PATCH_INPUT_SHAPE,
            patch_output_shape=PATCH_OUTPUT_SHAPE,
            slide_width=slide_width,
            slide_height=slide_height,
        )
        context_rgb, padding = _read_context_from_wsi(slide, geometry)
        windows = extract_windows_2x2(context_rgb, patch_input_shape=PATCH_INPUT_SHAPE)
        target_rgb = _read_target_patch_from_wsi(
            slide,
            int(metadata["x_level0"]),
            int(metadata["y_level0"]),
        )
        if context_rgb.shape != (CONTEXT_SIZE, CONTEXT_SIZE, 3):
            raise RuntimeError(f"Context expected 1536x1536x3, got {context_rgb.shape}.")
        if target_rgb.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            raise RuntimeError(f"Target patch expected 1024x1024x3, got {target_rgb.shape}.")

        input_preview_path = patch_output_dir / "input_preview.png"
        Image.fromarray(target_rgb).save(input_preview_path)
        context_artifacts: dict[str, str] = {}
        if save_context_artifacts:
            context_artifacts = _write_context_artifacts(patch_output_dir, context_rgb, windows)

        window_outputs, runtime_summary = _infer_context_windows(runtime, windows)
        stitched_labels, stitched_probabilities, coverage = _stitch_window_probabilities(
            window_outputs
        )
        class_pixel_counts, class_pixel_ratios = _class_distribution(stitched_labels)
        if sum(class_pixel_counts.values()) != PATCH_SIZE * PATCH_SIZE:
            raise RuntimeError("Class pixel counts do not sum to 1024*1024.")
        probability_summary = _probability_summary(stitched_probabilities, stitched_labels)

        labels_path = patch_output_dir / "prediction_labels_stitched_1024.npy"
        mask_path = patch_output_dir / "prediction_mask_stitched_1024.png"
        overlay_path = patch_output_dir / "prediction_overlay_stitched_1024.png"
        probabilities_path = patch_output_dir / "prediction_probabilities_stitched_1024.npz"
        manifest_path = patch_output_dir / "stitching_manifest.json"
        np.save(labels_path, stitched_labels)
        Image.fromarray(colorize_label_mask(stitched_labels)).save(mask_path)
        overlay = overlay_label_mask(target_rgb, stitched_labels, alpha=overlay_alpha)
        Image.fromarray(overlay).save(overlay_path)
        saved_probabilities_path = None
        if save_probabilities:
            np.savez_compressed(probabilities_path, probabilities=stitched_probabilities)
            saved_probabilities_path = str(probabilities_path)

        coverage_summary = {
            "pixels_without_coverage": int(np.sum(coverage == 0)),
            "min_coverage_count": int(np.min(coverage)),
            "max_coverage_count": int(np.max(coverage)),
            "mean_coverage_count": float(np.mean(coverage)),
        }
        windows_manifest = []
        for window_id in WINDOW_IDS:
            window_geometry = next(
                item for item in geometry["windows"] if item["window_id"] == window_id
            )
            windows_manifest.append(
                {
                    **window_geometry,
                    "runtime_seconds": window_outputs[window_id]["runtime_seconds"],
                    "prediction_shape": list(window_outputs[window_id]["labels"].shape),
                    "probability_shape": list(
                        window_outputs[window_id]["probabilities"].shape
                    ),
                    "prediction_source": window_outputs[window_id]["prediction_source"],
                    "probability_source": window_outputs[window_id]["probability_source"],
                }
            )

        manifest = {
            "status": "completed",
            "inference_strategy": INFERENCE_STRATEGY,
            "patch_id": patch_id,
            "axis_convention": AXIS_CONVENTION,
            "patch_input_shape": [PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE],
            "patch_output_shape": [PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE],
            "target_patch_size": PATCH_SIZE,
            "context_margin": CONTEXT_MARGIN,
            "context_shape": [CONTEXT_SIZE, CONTEXT_SIZE, 3],
            "geometry": geometry,
            "windows": windows_manifest,
            "padding": padding,
            "coverage": coverage_summary,
            "model_metadata": runtime.model_metadata,
            "selected_patch_png_validation_note": (
                "Selected PNG was validated for compatibility; context-stitch windows "
                "were read from the original WSI."
            ),
            "clinical_warning": CLINICAL_WARNING,
        }
        _write_json(manifest, manifest_path)

        summary.update(
            {
                "status": "completed",
                "image_width": selected_rgb.size[0],
                "image_height": selected_rgb.size[1],
                "prediction_shape": [PATCH_SIZE, PATCH_SIZE],
                "raw_prediction_shape": [PATCH_SIZE, PATCH_SIZE],
                "visualized_mask_shape": [PATCH_SIZE, PATCH_SIZE],
                "stitched_prediction_shape": [PATCH_SIZE, PATCH_SIZE],
                "resized_for_visualization": False,
                "num_classes_or_labels": len(class_pixel_counts),
                "unique_prediction_values": [
                    int(value) for value in np.unique(stitched_labels)[:100]
                ],
                "class_count_source": CLASS_COUNT_SOURCE,
                "class_pixel_counts": class_pixel_counts,
                "class_pixel_ratios": class_pixel_ratios,
                "raw_prediction_total_pixels": int(stitched_labels.size),
                "visualized_prediction_total_pixels": int(stitched_labels.size),
                "total_prediction_pixels": int(stitched_labels.size),
                "probability_summary": probability_summary,
                "runtime_seconds": time.perf_counter() - start,
                "mean_runtime_per_window": runtime_summary["mean_runtime_per_window"],
                "window_runtime_seconds": runtime_summary["per_window_runtime_seconds"],
                "padding": padding,
                "context_padding_used": bool(padding.get("context_padding_used")),
                "coverage": coverage_summary,
                "model_class": runtime.model_metadata.get("model_class"),
                "ioconfig_class": runtime.model_metadata.get("ioconfig_class"),
                "torch_version": runtime.model_metadata.get("torch_version"),
                "tiatoolbox_version": runtime.model_metadata.get("tiatoolbox_version"),
                "outputs": {
                    "input_preview": str(input_preview_path),
                    "prediction_labels_stitched_1024": str(labels_path),
                    "prediction_mask_stitched_1024": str(mask_path),
                    "prediction_overlay_stitched_1024": str(overlay_path),
                    "prediction_probabilities_stitched_1024": saved_probabilities_path,
                    "stitching_manifest": str(manifest_path),
                    "context_artifacts": context_artifacts,
                },
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - per-patch diagnostic
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["runtime_seconds"] = time.perf_counter() - start

    _write_json(summary, summary_path)
    return summary, summary_path
