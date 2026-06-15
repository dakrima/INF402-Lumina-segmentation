#!/usr/bin/env python
"""Dry-run checks for the future TIAToolbox BCSS baseline."""

from __future__ import annotations

import argparse
from pathlib import Path


TARGET_MODEL_NAME = "fcn_resnet50_unet-bcss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check TIAToolbox import and document the target BCSS baseline.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional path to a small image for future inference tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("TIAToolbox baseline check")
    print("=========================")
    print(f"Target model: {TARGET_MODEL_NAME}")
    print("This script does not download model weights or datasets.")

    try:
        import tiatoolbox  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"[WARN] TIAToolbox import failed: {exc}")
        tiatoolbox_available = False
    else:
        print("[ OK ] TIAToolbox import succeeded.")
        tiatoolbox_available = True

    if args.image_path is None:
        print("\nNo --image-path was provided.")
        print("Usage example:")
        print("  python scripts/02_test_tiatoolbox_model.py --image-path path/to/image.png")
        print("Real inference remains pending until weights and runtime behavior are verified.")
        return 0

    if not args.image_path.exists():
        print(f"[FAIL] Image path does not exist: {args.image_path}")
        return 1

    if not tiatoolbox_available:
        print("[FAIL] Cannot proceed with an image because TIAToolbox is unavailable.")
        return 1

    print(f"Image provided: {args.image_path}")
    print("Inference is not implemented yet to avoid implicit weight downloads.")
    print("Pending: verify TIAToolbox model loading and explicit weight handling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
