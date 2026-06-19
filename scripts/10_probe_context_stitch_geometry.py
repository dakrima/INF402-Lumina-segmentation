#!/usr/bin/env python
"""Probe context-stitch geometry for selected WSI patches."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.context_stitching import (  # noqa: E402
    AXIS_CONVENTION,
    CLINICAL_WARNING,
    WINDOW_IDS,
    compute_context_geometry,
    extract_windows_2x2,
    stitch_quadrants,
)
from src.inference.tiatoolbox_inference import run_inference_smoke_test  # noqa: E402
from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, SUPPORTED_DEVICES  # noqa: E402
from src.visualization.segmentation_overlay import colorize_label_mask, overlay_label_mask  # noqa: E402


PATCH_INPUT_SHAPE = 1024
PATCH_OUTPUT_SHAPE = 512
PATCH_SIZE = 1024
PADDING_RGB = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run technical probes for context-stitch 2x2 geometry. "
            "This does not diagnose, calculate RCB, validate clinical performance, "
            "or change the production inference pipeline."
        ),
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs"),
        help="Patch selection run directory containing selected_metadata.csv and selection_summary.json.",
    )
    parser.add_argument("--patch-index", type=int, default=0, help="Zero-based selected patch row index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/context_stitch_probe"),
        help="Output directory for probe artifacts.",
    )
    parser.add_argument("--run-synthetic", action="store_true", help="Run synthetic geometry probe.")
    parser.add_argument("--run-wsi-context", action="store_true", help="Read WSI context and save windows.")
    parser.add_argument(
        "--run-alignment-probe",
        action="store_true",
        help="Run model inference on four context windows and stitch raw outputs.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="cpu",
        help="Device for optional model alignment probe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output directory. Only safe repo output paths are cleared.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (ROOT_DIR / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = ROOT_DIR.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        resolved_root,
        resolved_root / "data",
        resolved_root / "outputs",
    }
    if resolved_output.exists() and any(resolved_output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {resolved_output}. "
            "Use --overwrite to regenerate probe outputs."
        )
    if overwrite and resolved_output.exists():
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite only clears output directories inside the repository.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _safe_int(value: object, field_name: str) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid integer metadata field: {field_name}={value!r}") from exc


def _first_non_empty(*values: object) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""


def _load_selected_patch(selection_dir: Path, patch_index: int) -> tuple[dict[str, str], dict[str, Any], Path, Path]:
    selected_metadata_path = selection_dir / "selected_metadata.csv"
    selection_summary_path = selection_dir / "selection_summary.json"
    if not selected_metadata_path.exists():
        raise FileNotFoundError(f"Missing selected metadata CSV: {selected_metadata_path}")
    if not selection_summary_path.exists():
        raise FileNotFoundError(f"Missing selection summary JSON: {selection_summary_path}")
    rows = _read_csv(selected_metadata_path)
    if patch_index < 0 or patch_index >= len(rows):
        raise IndexError(f"--patch-index {patch_index} outside selected metadata row range 0..{len(rows) - 1}")
    return rows[patch_index], _read_json(selection_summary_path), selected_metadata_path, selection_summary_path


def _patch_metadata(row: dict[str, str], selection_summary: dict[str, Any]) -> dict[str, Any]:
    wsi_path = _first_non_empty(row.get("source_wsi_path"), selection_summary.get("wsi_path"))
    if not wsi_path:
        raise ValueError("Missing WSI path in selected_metadata.csv and selection_summary.json.")
    x_level0 = _safe_int(row.get("x_level0"), "x_level0")
    y_level0 = _safe_int(row.get("y_level0"), "y_level0")
    patch_size = _safe_int(_first_non_empty(row.get("patch_size"), selection_summary.get("patch_size"), PATCH_SIZE), "patch_size")
    return {
        "patch_id": _first_non_empty(row.get("patch_id"), Path(row.get("filename", "")).stem),
        "filename": row.get("filename", ""),
        "wsi_path": str(Path(wsi_path).expanduser().resolve()),
        "x_level0": x_level0,
        "y_level0": y_level0,
        "patch_size": patch_size,
        "mpp_x": _first_non_empty(row.get("mpp_x"), selection_summary.get("mpp_x")),
        "mpp_y": _first_non_empty(row.get("mpp_y"), selection_summary.get("mpp_y")),
        "objective_power": _first_non_empty(row.get("objective_power"), selection_summary.get("objective_power")),
        "source_selection_method": _first_non_empty(row.get("selection_method"), selection_summary.get("selector")),
    }


def run_synthetic_probe(output_dir: Path) -> dict[str, Any]:
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    geometry = compute_context_geometry(x_level0=4096, y_level0=8192)

    context = np.zeros((1536, 1536, 3), dtype=np.uint8)
    y_grid, x_grid = np.indices((1536, 1536))
    context[..., 0] = (x_grid % 256).astype(np.uint8)
    context[..., 1] = (y_grid % 256).astype(np.uint8)
    context[..., 2] = 180

    simulated_outputs = {
        "window_00": np.full((512, 512), 1, dtype=np.int32),
        "window_01": np.full((512, 512), 2, dtype=np.int32),
        "window_10": np.full((512, 512), 3, dtype=np.int32),
        "window_11": np.full((512, 512), 4, dtype=np.int32),
    }
    stitched = stitch_quadrants(simulated_outputs)
    expected = (
        np.all(stitched[0:512, 0:512] == 1)
        and np.all(stitched[0:512, 512:1024] == 2)
        and np.all(stitched[512:1024, 0:512] == 3)
        and np.all(stitched[512:1024, 512:1024] == 4)
    )

    context_path = synthetic_dir / "synthetic_context.png"
    stitched_path = synthetic_dir / "synthetic_stitched.png"
    Image.fromarray(context).save(context_path)
    Image.fromarray(colorize_label_mask(stitched)).save(stitched_path)
    manifest = {
        "status": "completed" if expected else "failed",
        "test_type": "synthetic_geometry",
        "context_shape": [1536, 1536],
        "stitched_shape": list(stitched.shape),
        "axis_convention": AXIS_CONVENTION,
        "windows": geometry["windows"],
        "geometry_valid": bool(expected and stitched.shape == (1024, 1024)),
        "synthetic_context_png": str(context_path),
        "synthetic_stitched_png": str(stitched_path),
        "clinical_warning": CLINICAL_WARNING,
    }
    manifest_path = _write_json(manifest, synthetic_dir / "synthetic_manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _import_openslide() -> object:
    try:
        return importlib.import_module("openslide")
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic
        raise RuntimeError("Missing dependency: openslide. Activate the project Conda environment.") from exc


def _read_context_from_wsi(wsi_path: Path, geometry: dict[str, Any]) -> Image.Image:
    openslide_module = _import_openslide()
    slide = openslide_module.OpenSlide(str(wsi_path))
    try:
        context_size = int(geometry["context_size"])
        context = Image.new("RGB", (context_size, context_size), PADDING_RGB)
        read_width = int(geometry["read_width"])
        read_height = int(geometry["read_height"])
        if read_width <= 0 or read_height <= 0:
            raise ValueError("Requested context has no overlap with the WSI.")
        region = slide.read_region(
            (int(geometry["read_x0"]), int(geometry["read_y0"])),
            0,
            (read_width, read_height),
        ).convert("RGB")
        padding = geometry["padding"]
        context.paste(region, (int(padding["padding_left"]), int(padding["padding_top"])))
        return context
    finally:
        slide.close()


def run_wsi_context_probe(
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    selected_metadata_path: Path,
    selection_summary_path: Path,
) -> dict[str, Any]:
    wsi_dir = output_dir / "wsi_context"
    wsi_dir.mkdir(parents=True, exist_ok=True)
    wsi_path = Path(str(metadata["wsi_path"]))
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")

    openslide_module = _import_openslide()
    slide = openslide_module.OpenSlide(str(wsi_path))
    try:
        slide_width, slide_height = slide.dimensions
    finally:
        slide.close()

    geometry = compute_context_geometry(
        x_level0=int(metadata["x_level0"]),
        y_level0=int(metadata["y_level0"]),
        patch_size=int(metadata["patch_size"]),
        patch_input_shape=PATCH_INPUT_SHAPE,
        patch_output_shape=PATCH_OUTPUT_SHAPE,
        slide_width=slide_width,
        slide_height=slide_height,
    )
    context_image = _read_context_from_wsi(wsi_path=wsi_path, geometry=geometry)
    context_rgb = np.asarray(context_image)
    windows = extract_windows_2x2(context_rgb)
    margin = int(geometry["margin"])
    patch_size = int(geometry["target_patch_size"])
    target_rgb = context_rgb[margin:margin + patch_size, margin:margin + patch_size, :]

    context_path = wsi_dir / "context_preview.png"
    target_path = wsi_dir / "target_patch_preview.png"
    context_image.save(context_path)
    Image.fromarray(target_rgb).save(target_path)

    window_paths: dict[str, str] = {}
    for window_id in WINDOW_IDS:
        window_path = wsi_dir / f"{window_id}.png"
        Image.fromarray(windows[window_id]).save(window_path)
        window_paths[window_id] = str(window_path)

    manifest = {
        "status": "completed",
        "test_type": "wsi_context_read",
        "patch_id": metadata.get("patch_id"),
        "wsi_path": str(wsi_path),
        "target_x0": geometry["target_x0"],
        "target_y0": geometry["target_y0"],
        "target_x1": geometry["target_x1"],
        "target_y1": geometry["target_y1"],
        "target_patch_size": geometry["target_patch_size"],
        "context_x0_requested": geometry["context_x0_requested"],
        "context_y0_requested": geometry["context_y0_requested"],
        "context_shape": list(context_rgb.shape),
        "window_input_shape": [PATCH_INPUT_SHAPE, PATCH_INPUT_SHAPE, 3],
        "mpp_x": metadata.get("mpp_x"),
        "mpp_y": metadata.get("mpp_y"),
        "objective_power": metadata.get("objective_power"),
        "slide_width": slide_width,
        "slide_height": slide_height,
        "padding": geometry["padding"],
        "axis_convention": AXIS_CONVENTION,
        "windows": geometry["windows"],
        "context_preview_png": str(context_path),
        "target_patch_preview_png": str(target_path),
        "window_pngs": window_paths,
        "source_selected_metadata_csv": str(selected_metadata_path),
        "source_selection_summary_json": str(selection_summary_path),
        "clinical_warning": CLINICAL_WARNING,
    }
    manifest_path = _write_json(manifest, wsi_dir / "wsi_context_manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    manifest["_context_rgb"] = context_rgb
    manifest["_target_rgb"] = target_rgb
    manifest["_windows"] = windows
    return manifest


def run_alignment_probe(
    output_dir: Path,
    context_probe: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model_name: str,
    device: str,
) -> dict[str, Any]:
    alignment_dir = output_dir / "alignment_probe"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    target_rgb = np.asarray(context_probe["_target_rgb"])
    windows = context_probe["_windows"]
    labels_by_window: dict[str, np.ndarray] = {}
    window_output_shapes: dict[str, list[int]] = {}
    window_prediction_paths: dict[str, str] = {}
    per_window_summary_paths: dict[str, str] = {}

    for window_id in WINDOW_IDS:
        window_input_path = alignment_dir / f"{window_id}_input.png"
        Image.fromarray(windows[window_id]).save(window_input_path)
        window_output_dir = alignment_dir / "per_window" / window_id
        summary, summary_path = run_inference_smoke_test(
            image_path=window_input_path,
            output_dir=window_output_dir,
            root_dir=ROOT_DIR,
            model_name=model_name,
            requested_device=device,
            input_mode="patch",
            clear_output=True,
            selection_metadata={
                "mpp_x": metadata.get("mpp_x"),
                "mpp_y": metadata.get("mpp_y"),
                "source_wsi_path": metadata.get("wsi_path"),
                "x_level0": metadata.get("x_level0"),
                "y_level0": metadata.get("y_level0"),
                "patch_size": metadata.get("patch_size"),
            },
        )
        if summary.get("status") != "completed":
            raise RuntimeError(f"{window_id} inference failed: {summary.get('error')}")
        labels_path = Path(summary["outputs"]["prediction_labels_raw_npy"])
        labels = np.load(labels_path)
        if labels.shape != (PATCH_OUTPUT_SHAPE, PATCH_OUTPUT_SHAPE):
            raise ValueError(f"{window_id} output shape expected 512x512, got {labels.shape}.")
        labels_by_window[window_id] = labels
        window_output_shapes[window_id] = list(labels.shape)
        raw_prediction_copy = alignment_dir / f"{window_id}_prediction_raw.png"
        shutil.copy2(summary["outputs"]["prediction_mask_raw"], raw_prediction_copy)
        window_prediction_paths[window_id] = str(raw_prediction_copy)
        per_window_summary_paths[window_id] = str(summary_path)

    stitched = stitch_quadrants(labels_by_window)
    stitched_npy = alignment_dir / "stitched_prediction_1024.npy"
    stitched_png = alignment_dir / "stitched_prediction_1024.png"
    stitched_overlay = alignment_dir / "stitched_overlay_1024.png"
    np.save(stitched_npy, stitched)
    Image.fromarray(colorize_label_mask(stitched)).save(stitched_png)
    overlay = overlay_label_mask(Image.fromarray(target_rgb), stitched)
    Image.fromarray(overlay).save(stitched_overlay)

    manifest = {
        "status": "completed",
        "test_type": "alignment_probe_with_model",
        "patch_id": metadata.get("patch_id"),
        "model_name": model_name,
        "device": device,
        "hypothesis": (
            "The 512x512 output is treated as the central useful prediction for each "
            "1024x1024 input window."
        ),
        "hypothesis_confirmed": "visual_review_required",
        "reason": (
            "This probe verifies shapes and geometry. Final confirmation requires visual "
            "inspection and/or comparison against TIAToolbox WSI/region inference."
        ),
        "window_output_shapes": window_output_shapes,
        "stitched_prediction_shape": list(stitched.shape),
        "axis_convention": AXIS_CONVENTION,
        "window_prediction_raw_pngs": window_prediction_paths,
        "per_window_inference_summaries": per_window_summary_paths,
        "stitched_prediction_1024_npy": str(stitched_npy),
        "stitched_prediction_1024_png": str(stitched_png),
        "stitched_overlay_1024_png": str(stitched_overlay),
        "clinical_warning": CLINICAL_WARNING,
    }
    manifest_path = _write_json(manifest, alignment_dir / "alignment_probe_manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> int:
    args = parse_args()
    if not (args.run_synthetic or args.run_wsi_context or args.run_alignment_probe):
        print("[FAIL] Select at least one probe: --run-synthetic, --run-wsi-context, or --run-alignment-probe.")
        return 1

    selection_dir = _resolve_path(args.selection_dir)
    output_dir = _resolve_path(args.output_dir)

    try:
        _prepare_output_dir(output_dir=output_dir, overwrite=args.overwrite)
        run_summary: dict[str, Any] = {
            "status": "completed",
            "output_dir": str(output_dir),
            "selection_dir": str(selection_dir),
            "patch_index": args.patch_index,
            "run_synthetic": args.run_synthetic,
            "run_wsi_context": args.run_wsi_context,
            "run_alignment_probe": args.run_alignment_probe,
            "clinical_warning": CLINICAL_WARNING,
            "probes": {},
        }

        print("Context-stitch geometry probe")
        print("=============================")
        print(f"Selection dir: {selection_dir}")
        print(f"Patch index: {args.patch_index}")
        print(f"Output dir: {output_dir}")
        print(f"Clinical warning: {CLINICAL_WARNING}")

        if args.run_synthetic:
            synthetic_manifest = run_synthetic_probe(output_dir)
            run_summary["probes"]["synthetic"] = synthetic_manifest["manifest_path"]
            print(f"[OK] Synthetic probe: {synthetic_manifest['manifest_path']}")

        context_manifest: dict[str, Any] | None = None
        metadata: dict[str, Any] | None = None
        if args.run_wsi_context or args.run_alignment_probe:
            row, selection_summary, selected_metadata_path, selection_summary_path = _load_selected_patch(
                selection_dir=selection_dir,
                patch_index=args.patch_index,
            )
            metadata = _patch_metadata(row, selection_summary)
            context_manifest = run_wsi_context_probe(
                output_dir=output_dir,
                metadata=metadata,
                selected_metadata_path=selected_metadata_path,
                selection_summary_path=selection_summary_path,
            )
            run_summary["probes"]["wsi_context"] = context_manifest["manifest_path"]
            print(f"[OK] WSI context probe: {context_manifest['manifest_path']}")
            print(
                "Padding used: "
                f"{context_manifest['padding']['context_padding_used']} "
                f"({context_manifest['padding']})"
            )

        if args.run_alignment_probe:
            if context_manifest is None or metadata is None:
                raise RuntimeError("Alignment probe requires WSI context metadata.")
            alignment_manifest = run_alignment_probe(
                output_dir=output_dir,
                context_probe=context_manifest,
                metadata=metadata,
                model_name=args.model_name,
                device=args.device,
            )
            run_summary["probes"]["alignment_probe"] = alignment_manifest["manifest_path"]
            print(f"[OK] Alignment probe: {alignment_manifest['manifest_path']}")
            print(f"Stitched shape: {alignment_manifest['stitched_prediction_shape']}")
            print(f"Hypothesis confirmed: {alignment_manifest['hypothesis_confirmed']}")

        summary_path = _write_json(run_summary, output_dir / "probe_summary.json")
        print(f"[OK] Probe summary: {summary_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error": str(exc),
            "clinical_warning": CLINICAL_WARNING,
        }
        _write_json(failure, output_dir / "probe_summary.json")
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
