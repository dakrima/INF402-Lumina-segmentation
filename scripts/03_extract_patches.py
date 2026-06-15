#!/usr/bin/env python
"""Extract simple patches from small image files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.paths import PATCHES_DIR, ensure_directories  # noqa: E402
from src.patching.extract_patches import iter_patches  # noqa: E402
from src.patching.patch_filtering import compute_tissue_ratio  # noqa: E402


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected patches from a small image using a simple tissue filter.",
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-tissue-ratio", type=float, default=0.2)
    parser.add_argument("--background-threshold", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[FAIL] Unsupported image extension: {args.image_path.suffix}")
        print("Supported extensions: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return 1

    if not args.image_path.exists():
        print(f"[FAIL] Image path does not exist: {args.image_path}")
        return 1

    if not 0 <= args.min_tissue_ratio <= 1:
        print("[FAIL] --min-tissue-ratio must be between 0 and 1.")
        return 1

    ensure_directories()
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    print("Patch extraction for small images")
    print("=================================")
    print("This script does not process real WSI pyramids yet.")
    print("WSI support will be integrated later with OpenSlide/TIAToolbox.")
    print(f"Image: {args.image_path}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Minimum tissue ratio: {args.min_tissue_ratio}")

    metadata_path = PATCHES_DIR / "patches_metadata.csv"
    rows: list[dict[str, object]] = []
    selected_count = 0
    total_count = 0

    with Image.open(args.image_path) as image:
        rgb_image = image.convert("RGB")
        for patch_index, (x, y, patch) in enumerate(
            iter_patches(rgb_image, args.patch_size, args.stride),
            start=1,
        ):
            total_count += 1
            tissue_ratio = compute_tissue_ratio(
                patch,
                background_threshold=args.background_threshold,
            )
            selected = tissue_ratio >= args.min_tissue_ratio
            patch_id = f"{args.image_path.stem}_x{x}_y{y}_p{patch_index:05d}"
            patch_path = ""

            if selected:
                selected_count += 1
                output_path = PATCHES_DIR / f"{patch_id}.png"
                patch.save(output_path)
                patch_path = str(output_path)

            rows.append(
                {
                    "patch_id": patch_id,
                    "source_image": str(args.image_path),
                    "x": x,
                    "y": y,
                    "width": patch.width,
                    "height": patch.height,
                    "tissue_ratio": f"{tissue_ratio:.6f}",
                    "selected": selected,
                    "path": patch_path,
                }
            )

    fieldnames = [
        "patch_id",
        "source_image",
        "x",
        "y",
        "width",
        "height",
        "tissue_ratio",
        "selected",
        "path",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total patches evaluated: {total_count}")
    print(f"Selected patches saved: {selected_count}")
    print(f"Metadata: {metadata_path}")

    if total_count == 0:
        print("[WARN] No full-size patches were generated. Check image size and patch size.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
