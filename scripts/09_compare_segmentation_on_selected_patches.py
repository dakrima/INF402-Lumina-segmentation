#!/usr/bin/env python
"""Compare technical segmentation outputs from two selected-patch runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.segmentation_comparison import (  # noqa: E402
    CLINICAL_WARNING,
    PREVIEW_SOURCE_TO_FIELD,
    SegmentationComparisonConfig,
    compare_segmentation_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare technical segmentation outputs from two selected-patch runs. "
            "This does not diagnose, calculate RCB, validate clinical performance, "
            "or compare against ground truth."
        ),
    )
    parser.add_argument(
        "--baseline-seg-dir",
        type=Path,
        required=True,
        help="Segmentation directory generated for baseline selected patches.",
    )
    parser.add_argument(
        "--smart-seg-dir",
        type=Path,
        required=True,
        help="Segmentation directory generated for smart selector selected patches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where comparison artifacts will be written.",
    )
    parser.add_argument(
        "--max-preview-patches",
        type=int,
        default=8,
        help="Maximum completed patches shown per method in the preview image.",
    )
    parser.add_argument(
        "--preview-source",
        choices=sorted(PREVIEW_SOURCE_TO_FIELD),
        default="overlays_with_legend",
        help="Visual output type used in the side-by-side preview.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output directory. Only safe repo output paths are cleared.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SegmentationComparisonConfig(
        baseline_seg_dir=args.baseline_seg_dir,
        smart_seg_dir=args.smart_seg_dir,
        output_dir=args.output_dir,
        root_dir=ROOT_DIR,
        max_preview_patches=args.max_preview_patches,
        preview_source=args.preview_source,
        overwrite=args.overwrite,
    )

    print("Selected patch segmentation comparison")
    print("======================================")
    print(f"Baseline segmentation dir: {args.baseline_seg_dir}")
    print(f"Smart segmentation dir: {args.smart_seg_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Preview source: {args.preview_source}")
    print(f"Max preview patches: {args.max_preview_patches}")
    print(f"Clinical warning: {CLINICAL_WARNING}")

    try:
        summary = compare_segmentation_runs(config)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[FAIL] {exc}")
        return 1

    for warning in summary.get("validation_warnings", []):
        print(f"[WARN] {warning}")

    print("[OK] Segmentation comparison completed")
    print(f"Status: {summary['status']}")
    print(f"Baseline selector: {summary.get('baseline_selector')}")
    print(f"Smart selector: {summary.get('smart_selector')}")
    print(f"Baseline completed: {summary['num_baseline_completed']}")
    print(f"Smart completed: {summary['num_smart_completed']}")
    print(f"Baseline failed/skipped: {summary['num_baseline_failed']}/{summary['num_baseline_skipped']}")
    print(f"Smart failed/skipped: {summary['num_smart_failed']}/{summary['num_smart_skipped']}")
    print(f"Summary JSON: {summary['outputs']['segmentation_comparison_summary_json']}")
    print(f"Metrics CSV: {summary['outputs']['segmentation_comparison_metrics_csv']}")
    print(f"Class distribution CSV: {summary['outputs']['segmentation_class_distribution_csv']}")
    print(f"Patch rows CSV: {summary['outputs']['segmentation_patch_rows_csv']}")
    print(f"Preview PNG: {summary['outputs']['segmentation_comparison_preview_png']}")
    print(f"Notes MD: {summary['outputs']['segmentation_comparison_notes_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
