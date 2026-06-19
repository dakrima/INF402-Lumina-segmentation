#!/usr/bin/env python
"""Run technical segmentation over patches selected by a selector run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, SUPPORTED_DEVICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run technical semantic segmentation over already selected patches. "
            "This does not diagnose, calculate RCB, validate clinical performance, "
            "or compare selectors."
        ),
    )
    parser.add_argument(
        "--input-selection-dir",
        type=Path,
        required=True,
        help="Directory produced by scripts/06_select_wsi_patches.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where segmentation outputs will be written.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="auto",
        help="Device selection: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["patch", "wsi"],
        default="patch",
        help="Use patch for selected PNG patches, or wsi only for WSI-like inputs.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Opacity for the technical prediction overlay.",
    )
    parser.add_argument(
        "--limit-patches",
        type=int,
        default=None,
        help="Optional positive limit on valid patches processed in metadata order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output directory. Only safe repo output paths are cleared.",
    )
    parser.add_argument(
        "--strict-input-validation",
        action="store_true",
        help=(
            "Fail each patch before inference if technical input validation detects "
            "an incompatible patch shape, dtype, range, or channel count."
        ),
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save probability arrays per patch when TIAToolbox exposes them. Disabled by default.",
    )
    parser.add_argument(
        "--save-visual-labels-npy",
        action="store_true",
        help="Save visualization-resized label arrays per patch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from src.inference.selected_patch_segmentation import (
            CLINICAL_WARNING,
            SelectedPatchSegmentationConfig,
            segment_selected_patches,
        )
    except ModuleNotFoundError as exc:
        print(f"[FAIL] Missing Python dependency: {exc.name}")
        print("Activate the inf402-lumina-seg Conda/Mamba environment and retry.")
        return 1

    config = SelectedPatchSegmentationConfig(
        input_selection_dir=args.input_selection_dir,
        output_dir=args.output_dir,
        root_dir=ROOT_DIR,
        model_name=args.model_name,
        requested_device=args.device,
        input_mode=args.input_mode,
        overlay_alpha=args.overlay_alpha,
        limit_patches=args.limit_patches,
        overwrite=args.overwrite,
        strict_input_validation=args.strict_input_validation,
        save_probabilities=args.save_probabilities,
        save_visual_labels_npy=args.save_visual_labels_npy,
    )

    print("Selected patch segmentation")
    print("===========================")
    print(f"Input selection dir: {args.input_selection_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Requested device: {args.device}")
    print(f"Input mode: {args.input_mode}")
    print(f"Overlay alpha: {args.overlay_alpha}")
    print(f"Limit patches: {args.limit_patches}")
    print(f"Strict input validation: {args.strict_input_validation}")
    print(f"Save probabilities: {args.save_probabilities}")
    print(f"Save visual labels npy: {args.save_visual_labels_npy}")
    print(f"Clinical warning: {CLINICAL_WARNING}")

    try:
        summary = segment_selected_patches(config)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[FAIL] {exc}")
        return 1

    for warning in summary.get("warnings", []):
        print(f"[WARN] {warning}")

    print("[OK] Selected patch segmentation completed")
    print(f"Status: {summary['status']}")
    print(f"Selector: {summary.get('selector')}")
    print(f"Selection method: {summary.get('selection_method')}")
    print(f"Metadata rows: {summary['num_metadata_rows']}")
    print(f"Patches attempted: {summary['num_patches_attempted']}")
    print(f"Patches completed: {summary['num_patches_completed']}")
    print(f"Patches failed: {summary['num_patches_failed']}")
    print(f"Patches skipped: {summary['num_patches_skipped']}")
    print(f"Patch warnings: {summary.get('num_patch_warnings', 0)}")
    print(
        "Patches with resized visualization: "
        f"{summary.get('num_patches_with_resized_visualization', 0)}"
    )
    print(
        "Patches with probability summary: "
        f"{summary.get('num_patches_with_probability_summary', 0)}"
    )
    print(f"Input validation summary: {summary.get('input_validation_summary_json')}")
    print(f"Per-patch CSV: {summary['per_patch_segmentation_csv']}")
    print(f"Summary JSON: {summary['output_dir']}/inference_summary.json")
    print(f"Output dir: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
