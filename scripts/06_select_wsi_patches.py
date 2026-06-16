#!/usr/bin/env python
"""Select WSI patches using the INF402 Etapa 1 baseline selector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASELINE_SELECTOR_NAME = "baseline_tiatoolbox"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Etapa 1 WSI patch selection baseline. This performs technical "
            "patch selection only; it does not diagnose, calculate RCB, train models, "
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
        "--overwrite",
        action="store_true",
        help="Regenerate the selected output directory. Only safe repo output paths are cleared.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selector != BASELINE_SELECTOR_NAME:
        print(
            f"[FAIL] Selector '{args.selector}' todavía no está implementado. "
            f"Esta etapa solo soporta {BASELINE_SELECTOR_NAME}."
        )
        return 1

    try:
        from src.selection import BaselineSelectionConfig, run_baseline_selection
    except ModuleNotFoundError as exc:
        print(f"[FAIL] Missing Python dependency: {exc.name}")
        print("Activate the inf402-lumina-seg Conda/Mamba environment and retry.")
        return 1

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

    print("WSI patch selection - Etapa 1 baseline")
    print("======================================")
    print(f"Selector: {args.selector}")
    print(f"WSI path: {args.wsi_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Patch size / stride: {args.patch_size} / {args.stride}")
    print(f"Max patches: {args.max_patches}")
    print(f"Minimum tissue ratio: {args.min_tissue_ratio}")
    print(f"Seed: {args.seed}")
    print("Clinical warning: technical selection only; not diagnosis, not RCB.")

    try:
        summary = run_baseline_selection(config)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[FAIL] {exc}")
        return 1

    print("[OK] Patch selection completed")
    print(f"Slide dimensions: {summary['slide_width']} x {summary['slide_height']}")
    print(f"Candidates generated: {summary['num_candidates_generated']}")
    print(f"Thumbnail candidates passing mask: {summary['num_thumbnail_candidates_passing_mask']}")
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
