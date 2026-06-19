"""Controlled TIAToolbox inference smoke tests for small local images."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.inference.input_validation import validate_patch_input
from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, resolve_torch_device
from src.visualization.segmentation_overlay import (
    append_legend_to_image,
    colorize_label_mask,
    color_for_class_id,
    normalize_label_mask,
    overlay_label_mask,
    render_class_legend_image,
    resize_label_mask,
)


CLINICAL_WARNING = (
    "Technical segmentation/inference only. Not for diagnosis, not RCB, not clinical validation."
)
PREDICTION_RESOLUTION_NOTE = (
    "Class counts and ratios are computed on the raw prediction mask. "
    "Visual masks and overlays may be resized with nearest-neighbor interpolation "
    "only for visual inspection over the input patch."
)
CLASS_MAPPING_WARNING = (
    "Class names were not confirmed from TIAToolbox/BCSS metadata. "
    "Colors are visualization-only."
)
TIATOOLBOX_BCSS_MAPPING_SOURCE = "tiatoolbox_semantic_segmentation_example"
TIATOOLBOX_BCSS_MAPPING_WARNING = (
    "These class names correspond to the grouped TIAToolbox BCSS model output, "
    "not the raw BCSS ground-truth codes. In raw BCSS masks, 0 means "
    "outside_roi/don't care."
)
RAW_BCSS_ZERO_WARNING = (
    "In raw BCSS masks, 0 means outside_roi/don't care and should not be "
    "treated as other."
)
# TIAToolbox's semantic segmentation example spells class 2 as "Inflamatory".
# We use "Inflammatory" in project outputs for readability.
TIATOOLBOX_BCSS_OUTPUT_CLASS_NAMES = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflammatory",
    3: "Necrosis",
    4: "Others",
}
BCSS_RAW_GROUND_TRUTH_CLASS_NAMES = {
    0: "outside_roi",
    1: "tumor",
    2: "stroma",
    3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris",
    5: "glandular_secretions",
    6: "blood",
    7: "exclude",
    8: "metaplasia_NOS",
    9: "fat",
    10: "plasma_cells",
    11: "other_immune_infiltrate",
    12: "mucoid_material",
    13: "normal_acinus_or_duct",
    14: "lymphatics",
    15: "undetermined",
    16: "nerve",
    17: "skin_adnexa",
    18: "blood_vessel",
    19: "angioinvasion",
    20: "dcis",
    21: "other",
}
CLASS_MAPPING_REFERENCES = {
    "bcss_repository": "https://github.com/PathologyDataScience/BCSS",
    "bcss_raw_codes": (
        "https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv"
    ),
    "tiatoolbox_semantic_segmentation_example": (
        "https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/"
        "06-semantic-segmentation.html"
    ),
}
INPUT_MODES = {"patch", "wsi"}
CLASS_MAPPING_ATTR_HINTS = (
    "class",
    "label",
    "mapping",
    "map",
    "dataset",
    "name",
)


def _import_required_module(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostic smoke test
        raise RuntimeError(f"Missing dependency: {module_name} ({exc})") from exc


def _safe_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unavailable"))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def clear_output_dir_safely(output_dir: Path, root_dir: Path) -> None:
    """Clear a non-dangerous output directory inside the repository."""
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        resolved_root,
        resolved_root / "data",
        resolved_root / "outputs",
    }

    if not _is_relative_to(resolved_output, resolved_root):
        raise ValueError("--clear-output only supports output directories inside the repository.")
    if resolved_output in dangerous_paths:
        raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")

    resolved_output.mkdir(parents=True, exist_ok=True)
    for child in resolved_output.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _summary_base(
    model_name: str,
    image_path: Path,
    requested_device: str,
    input_mode: str,
    output_dir: Path,
    selection_metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    patch_mode = input_mode == "patch"
    return {
        "status": "failed",
        "model_name": model_name,
        "input_mode": input_mode,
        "patch_mode": patch_mode,
        "image_path": str(image_path),
        "image_width": None,
        "image_height": None,
        "patch_batch_shape": None,
        "requested_device": requested_device,
        "resolved_device": None,
        "tiatoolbox_version": None,
        "torch_version": None,
        "model_class": None,
        "ioconfig_class": None,
        "input_validation": {},
        "selection_metadata": selection_metadata or {},
        "prediction_shape": None,
        "raw_prediction_shape": None,
        "visualized_mask_shape": None,
        "resized_for_visualization": False,
        "num_classes_or_labels": None,
        "unique_prediction_values": [],
        "class_mapping_source": "unconfirmed",
        "class_mapping_warning": CLASS_MAPPING_WARNING,
        "raw_bcss_zero_warning": RAW_BCSS_ZERO_WARNING,
        "available_model_classes": {},
        "available_model_class_colors": {},
        "tiatoolbox_bcss_model_output_mapping": {},
        "bcss_raw_ground_truth_mapping": {
            str(class_id): class_name
            for class_id, class_name in sorted(BCSS_RAW_GROUND_TRUTH_CLASS_NAMES.items())
        },
        "class_mapping_references": CLASS_MAPPING_REFERENCES,
        "class_pixel_counts": {},
        "class_pixel_ratios": {},
        "class_count_source": "raw_prediction_mask",
        "raw_prediction_total_pixels": None,
        "visualized_prediction_total_pixels": None,
        "total_prediction_pixels": None,
        "prediction_resolution_note": PREDICTION_RESOLUTION_NOTE,
        "probability_summary": {
            "available": False,
            "reason": "Probability summary was not computed.",
        },
        "inference_config": {},
        "legend_json": None,
        "legend_png": None,
        "tiatoolbox_output": None,
        "tiatoolbox_output_type": None,
        "prediction_source": None,
        "outputs": {
            "input_preview": str(output_dir / "input_preview.png"),
            "prediction_mask_raw": str(output_dir / "prediction_mask_raw.png"),
            "prediction_mask_visual": str(output_dir / "prediction_mask_visual.png"),
            "prediction_mask": str(output_dir / "prediction_mask.png"),
            "prediction_labels_raw_npy": str(output_dir / "prediction_labels_raw.npy"),
            "prediction_labels_visual_npy": None,
            "prediction_probabilities_npz": None,
            "prediction_overlay": str(output_dir / "prediction_overlay.png"),
            "prediction_overlay_with_legend": str(
                output_dir / "prediction_overlay_with_legend.png"
            ),
            "legend_json": str(output_dir / "legend.json"),
            "legend_png": str(output_dir / "legend.png"),
        },
        "clinical_warning": CLINICAL_WARNING,
        "error": None,
        "suggested_next_step": None,
        "warnings": [],
    }


def write_inference_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    """Write an inference summary JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "inference_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def write_legend_json(legend: dict[str, Any], output_dir: Path) -> Path:
    """Write the technical class/color legend JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    legend_path = output_dir / "legend.json"
    legend_path.write_text(
        json.dumps(legend, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return legend_path


def _find_zarr_path(run_output: object, image_path: Path) -> Path:
    """Resolve a TIAToolbox SemanticSegmentor zarr path from run() output."""
    if isinstance(run_output, dict):
        possible_keys = [
            image_path,
            str(image_path),
            image_path.name,
            str(image_path.resolve()),
            image_path.resolve(),
        ]
        for key in possible_keys:
            if key in run_output:
                return Path(run_output[key])
        if len(run_output) == 1:
            return Path(next(iter(run_output.values())))
        raise RuntimeError(f"Could not resolve zarr path from SemanticSegmentor output keys: {list(run_output)}")

    if isinstance(run_output, (list, tuple)):
        if len(run_output) != 1:
            raise RuntimeError(f"Expected one output path, got {len(run_output)} entries.")
        return Path(run_output[0])

    return Path(run_output)


def _array_from_zarr_group(zarr_group: object, key: str) -> np.ndarray | None:
    if key not in list(zarr_group.keys()):
        return None
    return np.asarray(zarr_group[key])


def _probability_from_zarr(zarr_path: Path) -> tuple[np.ndarray | None, str | None, str | None]:
    zarr_module = _import_required_module("zarr")
    zarr_group = zarr_module.open(str(zarr_path), mode="r")
    probabilities = _array_from_zarr_group(zarr_group, "probabilities")
    if probabilities is None:
        return None, None, f"No probabilities array found in zarr output. Keys: {list(zarr_group.keys())}"
    return probabilities, "zarr.probabilities", None


def _prediction_from_zarr(zarr_path: Path) -> tuple[np.ndarray, str]:
    """Read predictions or probabilities from a TIAToolbox zarr output."""
    zarr_module = _import_required_module("zarr")
    zarr_group = zarr_module.open(str(zarr_path), mode="r")

    prediction = _array_from_zarr_group(zarr_group, "predictions")
    if prediction is not None:
        return normalize_label_mask(prediction), "predictions"

    probabilities = _array_from_zarr_group(zarr_group, "probabilities")
    if probabilities is not None:
        return normalize_label_mask(probabilities), "probabilities_argmax"

    keys = list(zarr_group.keys())
    raise RuntimeError(f"No 'predictions' or 'probabilities' array found in zarr output. Keys: {keys}")


def _safe_int(value: object) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _class_name_from_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("class_name", "label_name", "name", "label", "display_name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _coerce_class_mapping(value: object, model_name: str, depth: int = 0) -> dict[int, str]:
    """Return an id-to-name mapping only when explicit class names are present."""
    if depth > 2:
        return {}

    if isinstance(value, dict):
        direct_mapping: dict[int, str] = {}
        inverted_mapping: dict[int, str] = {}

        for key, item in value.items():
            class_id = _safe_int(key)
            class_name = _class_name_from_value(item)
            if class_id is not None and class_name is not None:
                direct_mapping[class_id] = class_name

            class_name = _class_name_from_value(key)
            class_id = _safe_int(item)
            if class_id is not None and class_name is not None:
                inverted_mapping[class_id] = class_name

        if direct_mapping:
            return direct_mapping
        if inverted_mapping:
            return inverted_mapping

        model_tokens = {model_name.lower(), model_name.replace("-", "_").lower(), "bcss"}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in model_tokens):
                nested_mapping = _coerce_class_mapping(item, model_name=model_name, depth=depth + 1)
                if nested_mapping:
                    return nested_mapping

    if isinstance(value, (list, tuple)):
        list_mapping: dict[int, str] = {}
        for index, item in enumerate(value):
            class_name = _class_name_from_value(item)
            class_id = index
            if isinstance(item, dict):
                for key in ("class_id", "label_id", "id", "index"):
                    explicit_id = _safe_int(item.get(key))
                    if explicit_id is not None:
                        class_id = explicit_id
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                explicit_id = _safe_int(item[0])
                explicit_name = _class_name_from_value(item[1])
                if explicit_id is not None and explicit_name is not None:
                    class_id = explicit_id
                    class_name = explicit_name

            if class_name is not None:
                list_mapping[class_id] = class_name
        if list_mapping:
            return list_mapping

    return {}


def _attribute_names_for_class_mapping(obj: object) -> list[str]:
    names: list[str] = []
    for name in dir(obj):
        lower_name = name.lower()
        if name.startswith("_"):
            continue
        if any(hint in lower_name for hint in CLASS_MAPPING_ATTR_HINTS):
            names.append(name)
    return names


def _class_mapping_from_object(obj: object | None, source: str, model_name: str) -> tuple[dict[int, str], str] | None:
    if obj is None:
        return None

    for attr_name in _attribute_names_for_class_mapping(obj):
        try:
            value = getattr(obj, attr_name)
        except Exception:  # noqa: BLE001 - diagnostic metadata discovery
            continue
        if callable(value):
            continue
        mapping = _coerce_class_mapping(value, model_name=model_name)
        if mapping:
            return mapping, f"{source}.{attr_name}"
    return None


def _class_mapping_from_tiatoolbox_modules(model_name: str) -> tuple[dict[int, str], str] | None:
    module_names = (
        "tiatoolbox.models",
        "tiatoolbox.models.architecture",
        "tiatoolbox.models.engine.io_config",
        "tiatoolbox.models.engine.semantic_segmentor",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - optional metadata discovery
            continue
        for attr_name in dir(module):
            lower_name = attr_name.lower()
            if attr_name.startswith("_"):
                continue
            if (
                "bcss" not in lower_name
                and "pretrain" not in lower_name
                and "model" not in lower_name
            ):
                continue
            try:
                value = getattr(module, attr_name)
            except Exception:  # noqa: BLE001 - optional metadata discovery
                continue
            if callable(value):
                continue
            mapping = _coerce_class_mapping(value, model_name=model_name)
            if mapping:
                return mapping, f"{module_name}.{attr_name}"
    return None


def discover_class_mapping(
    model: object | None,
    ioconfig: object | None,
    model_name: str,
) -> tuple[dict[int, str], str, str | None]:
    """Try to find explicit class names without inventing missing semantics."""
    if model_name == DEFAULT_MODEL_NAME:
        return (
            TIATOOLBOX_BCSS_OUTPUT_CLASS_NAMES.copy(),
            TIATOOLBOX_BCSS_MAPPING_SOURCE,
            TIATOOLBOX_BCSS_MAPPING_WARNING,
        )

    sources = (
        ("tiatoolbox_ioconfig", ioconfig),
        ("tiatoolbox_model", model),
    )
    for source, obj in sources:
        discovered = _class_mapping_from_object(obj=obj, source=source, model_name=model_name)
        if discovered is not None:
            mapping, mapping_source = discovered
            return mapping, mapping_source, None

    discovered = _class_mapping_from_tiatoolbox_modules(model_name=model_name)
    if discovered is not None:
        mapping, mapping_source = discovered
        return mapping, mapping_source, None

    return {}, "unconfirmed", CLASS_MAPPING_WARNING


def build_class_legend(
    label_mask: np.ndarray,
    model_name: str,
    class_names: dict[int, str],
    mapping_source: str,
    mapping_warning: str | None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, float]]:
    """Build a technical class/color legend from the predicted label mask."""
    label_mask = normalize_label_mask(label_mask)
    unique_values, counts = np.unique(label_mask, return_counts=True)
    total_pixels = int(label_mask.size)
    visible_class_ids = sorted(set(class_names) | {int(value) for value in unique_values})
    class_pixel_counts: dict[str, int] = {
        str(class_id): 0 for class_id in visible_class_ids
    }
    class_pixel_ratios: dict[str, float] = {
        str(class_id): 0.0 for class_id in visible_class_ids
    }
    classes: list[dict[str, Any]] = []

    observed_counts = {
        int(class_id_raw): int(count_raw)
        for class_id_raw, count_raw in zip(unique_values, counts, strict=True)
    }
    for class_id, count in observed_counts.items():
        class_pixel_counts[str(class_id)] = count
        class_pixel_ratios[str(class_id)] = count / total_pixels if total_pixels else 0.0

    for class_id in visible_class_ids:
        count = class_pixel_counts[str(class_id)]
        ratio = class_pixel_ratios[str(class_id)]
        class_name = class_names.get(class_id, "unconfirmed")
        if class_id not in class_names:
            class_name_status = "unconfirmed"
        elif mapping_source == TIATOOLBOX_BCSS_MAPPING_SOURCE:
            class_name_status = "confirmed_tiatoolbox_grouped_output"
        else:
            class_name_status = "confirmed"
        color_rgb = list(color_for_class_id(class_id))
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "class_name_status": class_name_status,
                "color_rgb": color_rgb,
                "pixel_count": count,
                "pixel_ratio": ratio,
            }
        )

    legend = {
        "status": "completed",
        "model_name": model_name,
        "mapping_source": mapping_source,
        "mapping_warning": mapping_warning,
        "raw_bcss_zero_warning": RAW_BCSS_ZERO_WARNING,
        "available_model_classes": {
            str(class_id): class_name for class_id, class_name in sorted(class_names.items())
        },
        "available_model_class_colors": {
            str(class_id): list(color_for_class_id(class_id))
            for class_id in sorted(class_names)
        },
        "tiatoolbox_bcss_model_output_mapping": (
            {
                str(class_id): class_name
                for class_id, class_name in sorted(TIATOOLBOX_BCSS_OUTPUT_CLASS_NAMES.items())
            }
            if model_name == DEFAULT_MODEL_NAME
            else {}
        ),
        "bcss_raw_ground_truth_mapping": {
            str(class_id): class_name
            for class_id, class_name in sorted(BCSS_RAW_GROUND_TRUTH_CLASS_NAMES.items())
        },
        "bcss_raw_ground_truth_zero": {
            "class_id": 0,
            "class_name": BCSS_RAW_GROUND_TRUTH_CLASS_NAMES[0],
            "warning": RAW_BCSS_ZERO_WARNING,
        },
        "class_mapping_references": CLASS_MAPPING_REFERENCES,
        "count_source": "raw_prediction_mask",
        "class_count_source": "raw_prediction_mask",
        "total_pixels": total_pixels,
        "total_prediction_pixels": total_pixels,
        "class_pixel_counts": class_pixel_counts,
        "class_pixel_ratios": class_pixel_ratios,
        "prediction_resolution_note": PREDICTION_RESOLUTION_NOTE,
        "classes": classes,
    }
    return legend, class_pixel_counts, class_pixel_ratios


def _to_numpy_array(value: object) -> np.ndarray:
    """Convert common TIAToolbox return values into a NumPy array."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "compute"):
        return np.asarray(value.compute())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _probability_from_dict(run_output: dict[Any, Any]) -> tuple[np.ndarray | None, str | None, str | None]:
    probability_keys = ("probabilities", "probability", "probs")
    parse_errors: list[str] = []

    for key in probability_keys:
        if key in run_output:
            try:
                return _to_numpy_array(run_output[key]), str(key), None
            except Exception as exc:  # noqa: BLE001 - preserve diagnostic
                parse_errors.append(f"{key}: {exc}")

    for key, value in run_output.items():
        if isinstance(value, dict):
            probabilities, source, reason = _probability_from_dict(value)
            if probabilities is not None:
                return probabilities, f"{key}.{source}", None
            if reason:
                parse_errors.append(f"{key}: {reason}")

    reason = "No probabilities key found in dict output."
    if parse_errors:
        reason = f"{reason} Parse errors: {'; '.join(parse_errors)}"
    return None, None, reason


def _probability_from_run_output(run_output: object) -> tuple[np.ndarray | None, str | None, str | None]:
    if isinstance(run_output, dict):
        return _probability_from_dict(run_output)

    if isinstance(run_output, (list, tuple)):
        if len(run_output) == 1:
            probabilities, source, reason = _probability_from_run_output(run_output[0])
            if probabilities is not None:
                return probabilities, f"list[0].{source}", None
            return None, None, reason
        return None, None, f"Could not inspect probabilities in list output with {len(run_output)} entries."

    if isinstance(run_output, (str, Path)):
        output_path = Path(run_output)
        if output_path.suffix == ".zarr" or output_path.is_dir():
            return _probability_from_zarr(output_path)
        return None, None, f"Unsupported probability path output from TIAToolbox: {output_path}"

    return None, None, f"No probability extraction rule for output type {type(run_output).__name__}."


def _probability_array_for_label_mask(
    probabilities: np.ndarray,
    label_mask: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    array = np.asarray(probabilities)
    array = np.squeeze(array)
    if array.ndim != 3:
        return None, f"Probability array must be 3D after squeeze, got shape {list(array.shape)}."

    if array.shape[:2] == label_mask.shape:
        return array.astype(np.float32, copy=False), None
    if array.shape[1:] == label_mask.shape:
        return np.moveaxis(array, 0, -1).astype(np.float32, copy=False), None
    return (
        None,
        "Probability spatial shape does not match raw prediction mask: "
        f"probabilities={list(array.shape)}, label_mask={list(label_mask.shape)}.",
    )


def build_probability_summary(
    run_output: object,
    label_mask: np.ndarray,
    *,
    output_dir: Path,
    save_probabilities: bool = False,
) -> dict[str, Any]:
    """Summarize prediction probabilities if TIAToolbox exposes them."""
    probabilities, source, reason = _probability_from_run_output(run_output)
    if probabilities is None:
        return {
            "available": False,
            "reason": reason or "Probabilities were not available in TIAToolbox output.",
        }

    probability_array, shape_reason = _probability_array_for_label_mask(
        probabilities=probabilities,
        label_mask=label_mask,
    )
    if probability_array is None:
        return {
            "available": False,
            "source": source,
            "reason": shape_reason,
            "raw_probability_shape": list(np.asarray(probabilities).shape),
        }

    max_probability = np.max(probability_array, axis=-1)
    mean_probability_by_predicted_class: dict[str, float] = {}
    for class_id in sorted(int(value) for value in np.unique(label_mask)):
        class_mask = label_mask == class_id
        if np.any(class_mask):
            mean_probability_by_predicted_class[str(class_id)] = float(
                np.mean(max_probability[class_mask])
            )

    probability_summary: dict[str, Any] = {
        "available": True,
        "source": source,
        "probability_shape": list(probability_array.shape),
        "mean_max_probability": float(np.mean(max_probability)),
        "median_max_probability": float(np.median(max_probability)),
        "min_max_probability": float(np.min(max_probability)),
        "max_max_probability": float(np.max(max_probability)),
        "mean_probability_by_predicted_class": mean_probability_by_predicted_class,
    }

    if save_probabilities:
        probability_path = output_dir / "prediction_probabilities.npz"
        np.savez_compressed(probability_path, probabilities=probability_array)
        probability_summary["saved_npz"] = str(probability_path)

    return probability_summary


def _prediction_from_dict(run_output: dict[Any, Any]) -> tuple[np.ndarray, str]:
    prediction_keys = ("predictions", "prediction", "labels", "label", "mask", "masks")
    probability_keys = ("probabilities", "probability", "probs", "logits", "semantic")
    parse_errors: list[str] = []

    for key in prediction_keys:
        if key in run_output:
            try:
                return normalize_label_mask(_to_numpy_array(run_output[key])), str(key)
            except Exception as exc:  # noqa: BLE001 - try alternate prediction payloads
                parse_errors.append(f"{key}: {exc}")

    for key in probability_keys:
        if key in run_output:
            try:
                return normalize_label_mask(_to_numpy_array(run_output[key])), f"{key}_argmax"
            except Exception as exc:  # noqa: BLE001 - report all attempted payloads
                parse_errors.append(f"{key}: {exc}")

    if len(run_output) == 1:
        only_key, only_value = next(iter(run_output.items()))
        label_mask, source = _prediction_from_run_output(only_value)
        return label_mask, f"{only_key}.{source}"

    for key, value in run_output.items():
        if isinstance(value, dict):
            try:
                label_mask, source = _prediction_from_dict(value)
            except RuntimeError:
                continue
            return label_mask, f"{key}.{source}"

    keys = ", ".join(str(key) for key in run_output.keys())
    details = f" Parse errors: {'; '.join(parse_errors)}" if parse_errors else ""
    raise RuntimeError(f"Could not find a usable mask in dict output. Keys: {keys}.{details}")


def _prediction_from_run_output(run_output: object) -> tuple[np.ndarray, str]:
    if isinstance(run_output, dict):
        return _prediction_from_dict(run_output)

    if isinstance(run_output, (list, tuple)):
        if len(run_output) == 1:
            label_mask, source = _prediction_from_run_output(run_output[0])
            return label_mask, f"list[0].{source}"
        try:
            return normalize_label_mask(_to_numpy_array(run_output)), "list_array"
        except Exception as exc:  # noqa: BLE001 - diagnostic output parsing
            raise RuntimeError(
                f"Could not interpret list output with {len(run_output)} entries."
            ) from exc

    if isinstance(run_output, (str, Path)):
        output_path = Path(run_output)
        if output_path.suffix == ".zarr" or output_path.is_dir():
            label_mask, source = _prediction_from_zarr(output_path)
            return label_mask, f"zarr.{source}"
        raise RuntimeError(f"Unsupported path output from TIAToolbox: {output_path}")

    return normalize_label_mask(_to_numpy_array(run_output)), "array"


def _build_segmentor(
    model_name: str,
    resolved_device: str,
) -> object:
    semantic_module = _import_required_module("tiatoolbox.models.engine.semantic_segmentor")
    segmentor_class = getattr(semantic_module, "SemanticSegmentor")
    return segmentor_class(
        model=model_name,
        batch_size=1,
        num_workers=0,
        device=resolved_device,
        verbose=True,
    )


def _run_segmentor_patch(
    model_name: str,
    patch_batch: np.ndarray,
    resolved_device: str,
) -> tuple[object, object]:
    segmentor = _build_segmentor(model_name=model_name, resolved_device=resolved_device)
    run_output = segmentor.run(
        patch_batch,
        patch_mode=True,
        output_type="dict",
        return_probabilities=True,
        device=resolved_device,
        verbose=True,
    )
    return segmentor, run_output


def _run_segmentor_wsi(
    model_name: str,
    image_path: Path,
    output_dir: Path,
    resolved_device: str,
) -> tuple[object, object]:
    segmentor = _build_segmentor(model_name=model_name, resolved_device=resolved_device)
    raw_output_dir = output_dir / "tiatoolbox_raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    run_output = segmentor.run(
        [str(image_path)],
        patch_mode=False,
        save_dir=str(raw_output_dir),
        overwrite=True,
        output_type="zarr",
        return_probabilities=True,
        auto_get_mask=False,
        device=resolved_device,
        verbose=True,
    )
    return segmentor, run_output


def run_inference_smoke_test(
    image_path: str | Path,
    output_dir: str | Path,
    root_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    requested_device: str = "auto",
    input_mode: str = "patch",
    overlay_alpha: float = 0.45,
    clear_output: bool = False,
    strict_input_validation: bool = False,
    selection_metadata: dict[str, object] | None = None,
    save_probabilities: bool = False,
    save_visual_labels_npy: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Run a TIAToolbox SemanticSegmentor smoke test and save visual outputs."""
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir)
    root_dir = Path(root_dir)
    if not output_dir.is_absolute():
        output_dir = (root_dir / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    if input_mode not in INPUT_MODES:
        supported = ", ".join(sorted(INPUT_MODES))
        raise ValueError(f"Unsupported input_mode '{input_mode}'. Choose one of: {supported}")

    summary = _summary_base(
        model_name=model_name,
        image_path=image_path,
        requested_device=requested_device,
        input_mode=input_mode,
        output_dir=output_dir,
        selection_metadata=selection_metadata,
    )
    summary["strict_input_validation"] = strict_input_validation
    summary["inference_config"] = {
        "model_name": model_name,
        "requested_device": requested_device,
        "resolved_device": None,
        "input_mode": input_mode,
        "patch_mode": input_mode == "patch",
        "batch_size": 1,
        "num_workers": 0,
        "return_probabilities": True,
        "output_type": "dict" if input_mode == "patch" else "zarr",
        "overlay_alpha": overlay_alpha,
        "strict_input_validation": strict_input_validation,
        "save_probabilities": save_probabilities,
        "save_visual_labels_npy": save_visual_labels_npy,
    }

    try:
        if clear_output:
            clear_output_dir_safely(output_dir=output_dir, root_dir=root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb_image, input_validation = validate_patch_input(
            image_path=image_path,
            selection_metadata=selection_metadata,
        )
        summary["input_validation"] = input_validation
        summary["warnings"].extend(str(warning) for warning in input_validation["warnings"])
        if strict_input_validation and input_validation["strict_validation_failed"]:
            raise ValueError(
                "Strict input validation failed; see input_validation in inference_summary.json."
            )

        image_width, image_height = rgb_image.size
        patch_array = np.array(rgb_image, dtype=np.uint8, copy=True)
        patch_batch = np.expand_dims(patch_array, axis=0)
        summary["image_width"] = image_width
        summary["image_height"] = image_height
        if input_mode == "patch":
            summary["patch_batch_shape"] = list(patch_batch.shape)

        input_preview_path = output_dir / "input_preview.png"
        rgb_image.save(input_preview_path)

        torch_module = _import_required_module("torch")
        tiatoolbox_module = _import_required_module("tiatoolbox")
        summary["torch_version"] = _safe_version(torch_module)
        summary["tiatoolbox_version"] = _safe_version(tiatoolbox_module)

        resolved_device = resolve_torch_device(torch_module, requested_device)
        summary["resolved_device"] = resolved_device
        summary["inference_config"]["resolved_device"] = resolved_device

        try:
            if input_mode == "patch":
                segmentor, run_output = _run_segmentor_patch(
                    model_name=model_name,
                    patch_batch=patch_batch,
                    resolved_device=resolved_device,
                )
            else:
                segmentor, run_output = _run_segmentor_wsi(
                    model_name=model_name,
                    image_path=image_path,
                    output_dir=output_dir,
                    resolved_device=resolved_device,
                )
        except Exception as exc:  # noqa: BLE001 - MPS fallback diagnostic
            if requested_device == "auto" and resolved_device == "mps":
                summary["warnings"].append(
                    "TIAToolbox inference failed on MPS; retrying on CPU. "
                    f"MPS error: {exc}"
                )
                resolved_device = "cpu"
                summary["resolved_device"] = resolved_device
                summary["inference_config"]["resolved_device"] = resolved_device
                if input_mode == "patch":
                    segmentor, run_output = _run_segmentor_patch(
                        model_name=model_name,
                        patch_batch=patch_batch,
                        resolved_device=resolved_device,
                    )
                else:
                    segmentor, run_output = _run_segmentor_wsi(
                        model_name=model_name,
                        image_path=image_path,
                        output_dir=output_dir,
                        resolved_device=resolved_device,
                    )
            else:
                raise

        summary["model_class"] = (
            f"{segmentor.model.__class__.__module__}.{segmentor.model.__class__.__name__}"
        )
        if getattr(segmentor, "ioconfig", None) is not None:
            summary["ioconfig_class"] = (
                f"{segmentor.ioconfig.__class__.__module__}.{segmentor.ioconfig.__class__.__name__}"
            )

        if input_mode == "wsi":
            zarr_path = _find_zarr_path(run_output=run_output, image_path=image_path)
            summary["tiatoolbox_output"] = str(zarr_path)
            summary["tiatoolbox_output_type"] = "zarr"
        else:
            summary["tiatoolbox_output"] = "in-memory"
            summary["tiatoolbox_output_type"] = type(run_output).__name__

        label_mask, prediction_source = _prediction_from_run_output(run_output)
        summary["prediction_source"] = prediction_source
        summary["prediction_shape"] = list(label_mask.shape)
        summary["raw_prediction_shape"] = list(label_mask.shape)

        unique_values = np.unique(label_mask)
        summary["unique_prediction_values"] = [int(value) for value in unique_values[:100]]
        summary["num_classes_or_labels"] = int(len(unique_values))

        class_names, mapping_source, mapping_warning = discover_class_mapping(
            model=getattr(segmentor, "model", None),
            ioconfig=getattr(segmentor, "ioconfig", None),
            model_name=model_name,
        )
        legend, class_pixel_counts, class_pixel_ratios = build_class_legend(
            label_mask=label_mask,
            model_name=model_name,
            class_names=class_names,
            mapping_source=mapping_source,
            mapping_warning=mapping_warning,
        )
        probability_summary = build_probability_summary(
            run_output=run_output,
            label_mask=label_mask,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )
        legend_png_path = output_dir / "legend.png"
        overlay_with_legend_path = output_dir / "prediction_overlay_with_legend.png"
        legend["legend_png"] = str(legend_png_path)
        legend["prediction_overlay_with_legend"] = str(overlay_with_legend_path)
        legend_path = write_legend_json(legend=legend, output_dir=output_dir)
        summary["class_mapping_source"] = mapping_source
        summary["class_mapping_warning"] = mapping_warning
        summary["raw_bcss_zero_warning"] = RAW_BCSS_ZERO_WARNING
        summary["available_model_classes"] = {
            str(class_id): class_name for class_id, class_name in sorted(class_names.items())
        }
        summary["available_model_class_colors"] = {
            str(class_id): list(color_for_class_id(class_id))
            for class_id in sorted(class_names)
        }
        if model_name == DEFAULT_MODEL_NAME:
            summary["tiatoolbox_bcss_model_output_mapping"] = {
                str(class_id): class_name
                for class_id, class_name in sorted(TIATOOLBOX_BCSS_OUTPUT_CLASS_NAMES.items())
            }
        summary["class_pixel_counts"] = class_pixel_counts
        summary["class_pixel_ratios"] = class_pixel_ratios
        summary["class_count_source"] = "raw_prediction_mask"
        summary["raw_prediction_total_pixels"] = int(label_mask.size)
        summary["total_prediction_pixels"] = int(label_mask.size)
        summary["probability_summary"] = probability_summary
        summary["legend_json"] = str(legend_path)
        summary["legend_png"] = str(legend_png_path)

        visual_mask = label_mask
        if visual_mask.shape != (image_height, image_width):
            summary["warnings"].append(
                "Prediction mask shape differs from input image; resized with nearest neighbor for visualization."
            )
            visual_mask = resize_label_mask(visual_mask, size=(image_width, image_height))
        summary["visualized_mask_shape"] = list(visual_mask.shape)
        summary["resized_for_visualization"] = list(label_mask.shape) != list(visual_mask.shape)
        summary["visualized_prediction_total_pixels"] = int(visual_mask.size)

        labels_raw_path = output_dir / "prediction_labels_raw.npy"
        np.save(labels_raw_path, label_mask)
        labels_visual_path: Path | None = None
        if save_visual_labels_npy:
            labels_visual_path = output_dir / "prediction_labels_visual.npy"
            np.save(labels_visual_path, visual_mask)

        raw_mask_rgb = colorize_label_mask(label_mask)
        raw_mask_path = output_dir / "prediction_mask_raw.png"
        Image.fromarray(raw_mask_rgb).save(raw_mask_path)

        mask_rgb = colorize_label_mask(visual_mask)
        visual_mask_path = output_dir / "prediction_mask_visual.png"
        Image.fromarray(mask_rgb).save(visual_mask_path)
        mask_path = output_dir / "prediction_mask.png"
        Image.fromarray(mask_rgb).save(mask_path)

        legend_image = render_class_legend_image(legend)
        legend_image.save(legend_png_path)

        overlay = overlay_label_mask(rgb_image, visual_mask, alpha=overlay_alpha)
        overlay_path = output_dir / "prediction_overlay.png"
        Image.fromarray(overlay).save(overlay_path)

        overlay_with_legend = append_legend_to_image(
            rgb_image=Image.fromarray(overlay),
            legend_image=legend_image,
        )
        overlay_with_legend.save(overlay_with_legend_path)

        summary["status"] = "completed"
        summary["outputs"] = {
            "input_preview": str(input_preview_path),
            "prediction_mask_raw": str(raw_mask_path),
            "prediction_mask_visual": str(visual_mask_path),
            "prediction_mask": str(mask_path),
            "prediction_mask_compatibility_note": (
                "prediction_mask.png is retained for compatibility and matches "
                "prediction_mask_visual.png."
            ),
            "prediction_labels_raw_npy": str(labels_raw_path),
            "prediction_labels_visual_npy": (
                str(labels_visual_path) if labels_visual_path is not None else None
            ),
            "prediction_probabilities_npz": probability_summary.get("saved_npz"),
            "prediction_overlay": str(overlay_path),
            "prediction_overlay_with_legend": str(overlay_with_legend_path),
            "legend_json": str(legend_path),
            "legend_png": str(legend_png_path),
        }
        summary["error"] = None
        summary["suggested_next_step"] = (
            "Inspect prediction_overlay.png visually, then decide whether a controlled "
            "patch-level or BCSS evaluation smoke test is needed."
        )
    except Exception as exc:  # noqa: BLE001 - smoke-test diagnostic
        summary["status"] = "failed"
        summary["error"] = str(exc)
        error_text = str(exc).lower()
        if input_mode == "wsi" and ("mpp" in error_text or "objective" in error_text):
            summary["suggested_next_step"] = (
                "This looks like a WSI scale metadata issue. For small tiles use "
                "--input-mode patch; for real WSI inputs provide scale metadata or "
                "explicit reader settings in a later WSI workflow."
            )
        else:
            summary["suggested_next_step"] = (
                "Run inside the inf402-lumina-seg environment; on macOS try prefixing the "
                "command with KMP_DUPLICATE_LIB_OK=TRUE if OpenMP/libomp aborts."
            )

    summary_path = write_inference_summary(summary=summary, output_dir=output_dir)
    return summary, summary_path
