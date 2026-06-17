#!/usr/bin/env python
"""Extract simple patches from small image files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.paths import PATCHES_DIR, ensure_directories  # noqa: E402
from src.patching.extract_patches import EDGE_POLICIES, iter_patches_with_metadata  # noqa: E402
from src.patching.patch_filtering import compute_tissue_ratio  # noqa: E402
from src.visualization.patch_preview import PatchBox, save_patch_selection_preview  # noqa: E402


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir.resolve()
    return (ROOT_DIR / output_dir).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _clear_output_dir_safely(output_dir: Path) -> None:
    """Clear only a non-dangerous output directory inside the repository."""
    resolved_output = output_dir.resolve()
    root_dir = ROOT_DIR.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        root_dir,
        root_dir / "data",
        root_dir / "outputs",
    }

    if not _is_relative_to(resolved_output, root_dir):
        raise ValueError(
            "--clear-output only supports output directories inside the repository."
        )
    if resolved_output in dangerous_paths:
        raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
    if resolved_output.parent == resolved_output:
        raise ValueError(f"Refusing to clear filesystem root: {resolved_output}")

    resolved_output.mkdir(parents=True, exist_ok=True)
    for child in resolved_output.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _coverage_extent(
    image_length: int,
    intervals: list[tuple[int, int]],
) -> tuple[int, bool]:
    """Return max covered extent and whether intervals cover the full axis."""
    if image_length <= 0 or not intervals:
        return 0, False

    sorted_intervals = sorted(
        (max(0, start), min(image_length, end)) for start, end in intervals
    )
    max_extent = max(end for _, end in sorted_intervals)
    covered_until = 0
    has_full_coverage = sorted_intervals[0][0] == 0

    for start, end in sorted_intervals:
        if start > covered_until:
            has_full_coverage = False
            break
        covered_until = max(covered_until, end)
        if covered_until >= image_length:
            break

    return max_extent, has_full_coverage and covered_until >= image_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected patches from a small image using a simple tissue filter.",
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-tissue-ratio", type=float, default=0.2)
    parser.add_argument("--background-threshold", type=int, default=220)
    parser.add_argument(
        "--edge-policy",
        choices=sorted(EDGE_POLICIES),
        default="drop",
        help=(
            "How to handle image borders: drop ignores incomplete borders; "
            "overlap shifts the last full patch to cover borders; pad fills "
            "border patches with white background."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATCHES_DIR,
        help="Directory where patch images, metadata, summary and preview are saved.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Safely clear the selected output directory before generating patches.",
    )
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help="Save rejected patch images under rejected/. Metadata is always recorded.",
    )
    parser.add_argument(
        "--summary-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.json. Enabled by default; use --no-summary-json to disable.",
    )
    parser.add_argument(
        "--preview-image",
        action="store_true",
        help="Write patch_selection_preview.png with selected/rejected rectangles.",
    )
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
    output_dir = _resolve_output_dir(args.output_dir)
    if args.clear_output:
        try:
            _clear_output_dir_safely(output_dir)
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 1

    selected_dir = output_dir / "selected"
    rejected_dir = output_dir / "rejected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    print("Patch extraction for small images")
    print("=================================")
    print("This script does not process real WSI pyramids yet.")
    print("WSI support will be integrated later with OpenSlide/TIAToolbox.")
    print(f"Image: {args.image_path}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Minimum tissue ratio: {args.min_tissue_ratio}")
    print(f"Edge policy: {args.edge_policy}")
    print(f"Output directory: {output_dir}")

    metadata_path = output_dir / "patches_metadata.csv"
    summary_path = output_dir / "summary.json"
    preview_path = output_dir / "patch_selection_preview.png"
    rows: list[dict[str, object]] = []
    preview_boxes: list[PatchBox] = []
    selected_count = 0
    total_count = 0
    padded_count = 0
    image_width = 0
    image_height = 0
    x_intervals: list[tuple[int, int]] = []
    y_intervals: list[tuple[int, int]] = []

    with Image.open(args.image_path) as image:
        rgb_image = image.convert("RGB")
        image_width, image_height = rgb_image.size
        for patch_index, extracted_patch in enumerate(
            iter_patches_with_metadata(
                rgb_image,
                args.patch_size,
                args.stride,
                edge_policy=args.edge_policy,
            ),
            start=1,
        ):
            x = extracted_patch.x
            y = extracted_patch.y
            patch = extracted_patch.image
            total_count += 1
            if extracted_patch.padded:
                padded_count += 1
            tissue_ratio = compute_tissue_ratio(
                patch,
                background_threshold=args.background_threshold,
            )
            selected = tissue_ratio >= args.min_tissue_ratio
            patch_id = f"{args.image_path.stem}_x{x}_y{y}_p{patch_index:05d}"
            patch_path = ""
            saved = False
            split = "selected" if selected else "rejected"

            if selected:
                selected_count += 1
                output_path = selected_dir / f"{patch_id}.png"
                patch.save(output_path)
                patch_path = str(output_path)
                saved = True
            elif args.save_rejected:
                output_path = rejected_dir / f"{patch_id}.png"
                patch.save(output_path)
                patch_path = str(output_path)
                saved = True

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
                    "saved": saved,
                    "split": split,
                    "edge_policy": args.edge_policy,
                    "padded": extracted_patch.padded,
                    "original_width": extracted_patch.original_width,
                    "original_height": extracted_patch.original_height,
                }
            )
            preview_boxes.append(
                PatchBox(
                    x=x,
                    y=y,
                    width=patch.width,
                    height=patch.height,
                    selected=selected,
                )
            )
            x_intervals.append((x, x + extracted_patch.original_width))
            y_intervals.append((y, y + extracted_patch.original_height))

        if args.preview_image:
            save_patch_selection_preview(
                rgb_image=rgb_image,
                patches=preview_boxes,
                output_path=preview_path,
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
        "saved",
        "split",
        "edge_policy",
        "padded",
        "original_width",
        "original_height",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rejected_count = total_count - selected_count
    selected_ratio = selected_count / total_count if total_count else 0.0
    coverage_width, covers_width = _coverage_extent(image_width, x_intervals)
    coverage_height, covers_height = _coverage_extent(image_height, y_intervals)
    covers_full_image = covers_width and covers_height
    if args.summary_json:
        summary = {
            "source_image": str(args.image_path),
            "patch_size": args.patch_size,
            "stride": args.stride,
            "min_tissue_ratio": args.min_tissue_ratio,
            "edge_policy": args.edge_policy,
            "image_width": image_width,
            "image_height": image_height,
            "coverage_width": coverage_width,
            "coverage_height": coverage_height,
            "covers_full_image": covers_full_image,
            "padded_patches": padded_count,
            "total_patches": total_count,
            "selected_patches": selected_count,
            "rejected_patches": rejected_count,
            "selected_ratio": round(selected_ratio, 4),
            "output_dir": str(output_dir),
            "metadata_csv": str(metadata_path),
            "preview_image": str(preview_path) if args.preview_image else "",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Total patches evaluated: {total_count}")
    print(f"Selected patches: {selected_count}")
    print(f"Rejected patches: {rejected_count}")
    print(f"Selected ratio: {selected_ratio:.4f}")
    print(f"Coverage: {coverage_width}x{coverage_height} of {image_width}x{image_height}")
    print(f"Covers full image: {covers_full_image}")
    print(f"Padded patches: {padded_count}")
    print(f"Metadata: {metadata_path}")
    if args.summary_json:
        print(f"Summary: {summary_path}")
    if args.preview_image:
        print(f"Preview: {preview_path}")
    print(f"Selected patches dir: {selected_dir}")
    if args.save_rejected:
        print(f"Rejected patches dir: {rejected_dir}")

    if total_count == 0:
        print("[WARN] No full-size patches were generated. Check image size and patch size.")
        if args.edge_policy == "overlap":
            print("[WARN] edge-policy overlap requires image dimensions >= patch size. Use pad for smaller images.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
