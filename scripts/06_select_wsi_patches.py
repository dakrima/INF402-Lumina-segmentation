#!/usr/bin/env python
"""Select WSI patches using INF402 baseline or smart selectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASELINE_SELECTOR_NAME = "baseline_tiatoolbox"
SMART_SELECTOR_NAME = "smart_tissue_nuclei_v1"
SUPPORTED_SELECTORS = (BASELINE_SELECTOR_NAME, SMART_SELECTOR_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run WSI patch selection for INF402. This performs technical patch "
            "selection only; it does not diagnose, calculate RCB, train models, "
            "or run semantic segmentation."
        ),
    )
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector", default=BASELINE_SELECTOR_NAME)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-patches", type=int, default=16)
    parser.add_argument("--min-tissue-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumbnail-max-size", type=int, default=2048)
    parser.add_argument(
        "--max-candidates-to-score",
        type=int,
        default=300,
        help=(
            "For smart_tissue_nuclei_v1 only. 0 scores all thumbnail-filtered "
            "candidates; N > 0 scores at most N candidates after seeded shuffle."
        ),
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=256,
        help="For smart_tissue_nuclei_v1 only. Downsampled patch size used for features.",
    )
    parser.add_argument(
        "--lambda-spatial",
        type=float,
        default=0.15,
        help="For smart_tissue_nuclei_v1 only. Spatial redundancy penalty weight.",
    )
    parser.add_argument(
        "--min-distance-level0",
        type=int,
        default=None,
        help=(
            "For smart_tissue_nuclei_v1 only. Distance scale for spatial penalty. "
            "Defaults to --patch-size."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate the selected output directory. Only safe repo output paths are cleared.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selector not in SUPPORTED_SELECTORS:
        print(
            f"[FAIL] Selector '{args.selector}' no está implementado. "
            "Selectores soportados: " + ", ".join(SUPPORTED_SELECTORS)
        )
        return 1

    try:
        from src.selection import (
            BaselineSelectionConfig,
            SmartTissueNucleiConfig,
            run_baseline_selection,
            run_smart_tissue_nuclei_selection,
        )
    except ModuleNotFoundError as exc:
        print(f"[FAIL] Missing Python dependency: {exc.name}")
        print("Activate the inf402-lumina-seg Conda/Mamba environment and retry.")
        return 1

    if args.selector == BASELINE_SELECTOR_NAME:
        config = BaselineSelectionConfig(
            wsi_path=args.wsi_path,
            output_dir=args.output_dir,
            root_dir=ROOT_DIR,
            selector=args.selector,
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            min_tissue_ratio=args.min_tissue_ratio,
            seed=args.seed,
            thumbnail_max_size=args.thumbnail_max_size,
            overwrite=args.overwrite,
        )
        runner = run_baseline_selection
    else:
        config = SmartTissueNucleiConfig(
            wsi_path=args.wsi_path,
            output_dir=args.output_dir,
            root_dir=ROOT_DIR,
            selector=args.selector,
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            min_tissue_ratio=args.min_tissue_ratio,
            seed=args.seed,
            thumbnail_max_size=args.thumbnail_max_size,
            overwrite=args.overwrite,
            max_candidates_to_score=args.max_candidates_to_score,
            feature_size=args.feature_size,
            lambda_spatial=args.lambda_spatial,
            min_distance_level0=args.min_distance_level0,
        )
        runner = run_smart_tissue_nuclei_selection

    print("WSI patch selection")
    print("===================")
    print(f"Selector: {args.selector}")
    print(f"WSI path: {args.wsi_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Patch size / stride: {args.patch_size} / {args.stride}")
    print(f"Max patches: {args.max_patches}")
    print(f"Minimum tissue ratio: {args.min_tissue_ratio}")
    print(f"Seed: {args.seed}")
    if args.selector == SMART_SELECTOR_NAME:
        print(f"Max candidates to score: {args.max_candidates_to_score}")
        print(f"Feature size: {args.feature_size}")
        print(f"Lambda spatial: {args.lambda_spatial}")
        print(f"Minimum distance level 0: {args.min_distance_level0 or args.patch_size}")
    print("Clinical warning: technical selection only; not diagnosis, not RCB.")

    try:
        summary = runner(config)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[FAIL] {exc}")
        return 1

    print("[OK] Patch selection completed")
    print(f"Slide dimensions: {summary['slide_width']} x {summary['slide_height']}")
    print(f"Candidates generated: {summary['num_candidates_generated']}")
    print(f"Thumbnail candidates passing mask: {summary['num_thumbnail_candidates_passing_mask']}")
    if "num_candidates_scored" in summary:
        print(f"Candidates scored: {summary['num_candidates_scored']}")
    print(f"Candidates evaluated: {summary['num_candidates_evaluated']}")
    print(f"Selected patches: {summary['num_selected']}")
    print(f"Selected dir: {summary['selected_dir']}")
    print(f"Candidate metadata: {summary['candidate_metadata_csv']}")
    print(f"Selected metadata: {summary['selected_metadata_csv']}")
    print(f"Summary JSON: {summary['output_dir']}/selection_summary.json")
    print(f"Method config: {summary['method_config_json']}")
    print(f"Preview image: {summary['preview_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
