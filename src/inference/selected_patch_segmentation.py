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


CLINICAL_WARNING = (
    "Technical segmentation over selected patches only. Not for diagnosis, "
    "not RCB, not clinical validation."
)
PREDICTION_RESOLUTION_NOTE = (
    "The raw prediction mask may have a different resolution than the input patch. "
    "For visualization, masks are resized to the patch size using nearest-neighbor "
    "interpolation to preserve discrete class labels. Pixel counts refer to the raw "
    "prediction resolution unless explicitly stated otherwise."
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
    "mask_path",
    "overlay_path",
    "overlay_with_legend_path",
    "input_preview_path",
    "patch_inference_summary_path",
    "input_image_shape",
    "raw_prediction_shape",
    "resized_for_visualization",
    "num_patch_warnings",
    "patch_warnings",
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
        "mask_path": "",
        "overlay_path": "",
        "overlay_with_legend_path": "",
        "input_preview_path": "",
        "patch_inference_summary_path": "",
        "input_image_shape": "",
        "raw_prediction_shape": "",
        "resized_for_visualization": "false",
        "num_patch_warnings": 0,
        "patch_warnings": "[]",
    }


def _copy_completed_outputs(
    patch_id: str,
    patch_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    outputs = patch_summary.get("outputs", {})
    copy_specs = {
        "prediction_mask": (
            output_dir / "masks" / f"{patch_id}__prediction_mask.png"
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
        "mask_path": copied_outputs.get("prediction_mask", ""),
        "overlay_path": copied_outputs.get("prediction_overlay", ""),
        "overlay_with_legend_path": copied_outputs.get(
            "prediction_overlay_with_legend",
            "",
        ),
        "input_preview_path": copied_outputs.get("input_preview", ""),
        "patch_inference_summary_path": str(patch_summary_path),
        "input_image_shape": _json_cell(_shape_from_image_size(patch_summary)),
        "raw_prediction_shape": _json_cell(patch_summary.get("prediction_shape")),
        "resized_for_visualization": (
            "true" if _summary_was_resized_for_visualization(patch_summary) else "false"
        ),
        "num_patch_warnings": len(patch_warnings),
        "patch_warnings": _json_cell(patch_warnings),
    }


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
    unique_patch_warnings: set[str] = set()

    for index, row in enumerate(selected_rows, start=1):
        filename = row.get("filename", "").strip()
        if not filename:
            skipped += 1
            warnings.append(f"Row {index} skipped: missing filename.")
            result_rows.append(_empty_result_row(row, patch_id="", status="skipped", error="missing filename"))
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
            )
            patch_warnings = [str(warning) for warning in patch_summary.get("warnings") or []]
            if patch_warnings:
                num_patches_with_warnings += 1
                num_patch_warnings += len(patch_warnings)
                unique_patch_warnings.update(patch_warnings)
            if _summary_was_resized_for_visualization(patch_summary):
                num_patches_with_resized_visualization += 1
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
                )
                | {"patch_inference_summary_path": str(patch_summary_path)}
            )
            warnings.append(f"Patch {patch_id} failed: {exc}")

    _write_csv(result_rows, per_patch_csv_path, PER_PATCH_SEGMENTATION_FIELDS)

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
        "prediction_resolution_note": PREDICTION_RESOLUTION_NOTE,
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
        "selection_summary_path": str(selection_summary_path),
        "selection_method_config_path": str(method_config_path),
        "per_patch_segmentation_csv": str(per_patch_csv_path),
        "warnings": warnings,
        "clinical_warning": CLINICAL_WARNING,
    }
    _write_json(summary, global_summary_path)
    return summary
