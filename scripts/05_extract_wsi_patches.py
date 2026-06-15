#!/usr/bin/env python
"""Extract reproducible level-0 patches from a WSI using OpenSlide."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.wsi_patch_extraction import (  # noqa: E402
    SUPPORTED_WSI_EXTENSIONS,
    WsiPatchExtractionConfig,
    extract_wsi_patches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a small, reproducible set of level-0 WSI patches using OpenSlide. "
            "This does not run inference, evaluate quality, diagnose, or calculate RCB."
        ),
    )
    parser.add_argument("--wsi-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wsi_patches/test_slide"),
    )
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--max-patches", type=int, default=8)
    parser.add_argument("--min-tissue-ratio", type=float, default=0.2)
    parser.add_argument("--thumbnail-size", type=int, default=2048)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Safely clear only the selected output directory before extraction.",
    )
    parser.add_argument(
        "--preview-image",
        action="store_true",
        help="Save patch_selection_preview.png with selected/rejected rectangles.",
    )
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help="Save rejected patch images under rejected/. Metadata is always recorded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when shuffling thumbnail tissue candidates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extension = args.wsi_path.suffix.lower()
    if extension not in SUPPORTED_WSI_EXTENSIONS:
        print(f"[FAIL] Unsupported WSI extension: {extension}")
        print("Supported extensions: " + ", ".join(sorted(SUPPORTED_WSI_EXTENSIONS)))
        return 1

    config = WsiPatchExtractionConfig(
        wsi_path=args.wsi_path,
        output_dir=args.output_dir,
        root_dir=ROOT_DIR,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        min_tissue_ratio=args.min_tissue_ratio,
        thumbnail_size=args.thumbnail_size,
        clear_output=args.clear_output,
        preview_image=args.preview_image,
        save_rejected=args.save_rejected,
        seed=args.seed,
    )

    print("OpenSlide WSI patch extraction")
    print("==============================")
    print(f"WSI path: {args.wsi_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Max patches: {args.max_patches}")
    print(f"Minimum tissue ratio: {args.min_tissue_ratio}")
    print(f"Thumbnail size: {args.thumbnail_size}")
    print(f"Seed: {args.seed}")
    print("Clinical warning: technical patch extraction only; not diagnosis, not RCB.")

    summary, summary_path = extract_wsi_patches(config)

    print(f"Status: {summary['status']}")
    print(f"Slide dimensions: {summary.get('slide_width')} x {summary.get('slide_height')}")
    print(f"Level count: {summary.get('level_count')}")
    print(f"Objective power: {summary.get('objective_power')}")
    print(f"MPP X/Y: {summary.get('mpp_x')} / {summary.get('mpp_y')}")
    print(f"Grid candidates: {summary.get('num_grid_candidates')}")
    print(f"Thumbnail candidates: {summary.get('num_thumbnail_candidates')}")
    print(f"Candidates evaluated: {summary.get('num_candidates_evaluated')}")
    print(f"Selected patches: {summary.get('num_selected')}")
    print(f"Rejected patches: {summary.get('num_rejected')}")
    print(f"Metadata CSV: {summary.get('metadata_csv')}")
    print(f"Preview image: {summary.get('preview_image')}")
    print(f"Summary JSON: {summary_path}")

    if summary["status"] != "completed":
        print(f"Error: {summary.get('error')}")
        print(f"Suggested next step: {summary.get('suggested_next_step')}")
        return 1

    print("Next step example:")
    print(
        "  KMP_DUPLICATE_LIB_OK=TRUE python scripts/04_run_inference.py "
        "--image-path <selected_patch.png> --model-name fcn_resnet50_unet-bcss "
        "--device cpu --input-mode patch --output-dir outputs/inference_smoke/test_wsi_patch_0000 "
        "--clear-output"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
