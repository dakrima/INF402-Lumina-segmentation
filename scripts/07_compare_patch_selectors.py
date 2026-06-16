#!/usr/bin/env python
"""Compare baseline and smart WSI patch selector outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare patch selector output folders. This does not run segmentation, "
            "train models, diagnose, or calculate RCB."
        ),
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--smart-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument(
        "--recompute-selected-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute shared features on selected PNGs only. Enabled by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate the comparison output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from src.selection.comparison import ComparisonConfig, compare_patch_selectors
    except ModuleNotFoundError as exc:
        print(f"[FAIL] Missing Python dependency: {exc.name}")
        print("Activate the inf402-lumina-seg Conda/Mamba environment and retry.")
        return 1

    config = ComparisonConfig(
        baseline_dir=args.baseline_dir,
        smart_dir=args.smart_dir,
        output_dir=args.output_dir,
        root_dir=ROOT_DIR,
        feature_size=args.feature_size,
        overwrite=args.overwrite,
        recompute_selected_features=args.recompute_selected_features,
    )

    print("Patch selector comparison")
    print("=========================")
    print(f"Baseline dir: {args.baseline_dir}")
    print(f"Smart dir: {args.smart_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Feature size: {args.feature_size}")
    print(f"Recompute selected features: {args.recompute_selected_features}")
    print("Clinical warning: technical comparison only; not diagnosis, not RCB.")

    try:
        summary = compare_patch_selectors(config)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[FAIL] {exc}")
        return 1

    warnings = summary.get("validation_warnings", [])
    if warnings:
        print("[WARN] Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    overlap = summary["overlap_metrics"]
    print("[OK] Comparison completed")
    print(f"Overlap selected: {overlap['num_overlap_selected']}")
    print(f"Jaccard selected: {overlap['jaccard_selected']}")
    print(f"Summary JSON: {summary['outputs']['comparison_summary_json']}")
    print(f"Metrics CSV: {summary['outputs']['comparison_metrics_csv']}")
    print(f"Selected overlap CSV: {summary['outputs']['selected_overlap_csv']}")
    print(f"Selected patches CSV: {summary['outputs']['comparison_selected_patches_csv']}")
    print(f"Preview image: {summary['outputs']['comparison_preview_png']}")
    print(
        "Selected-only preview image: "
        f"{summary['outputs']['comparison_preview_selected_only_png']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
