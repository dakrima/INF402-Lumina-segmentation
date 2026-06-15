#!/usr/bin/env python
"""Smoke-test loading the TIAToolbox BCSS pretrained segmentation model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.tiatoolbox_bcss import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_STATUS_JSON,
    SUPPORTED_DEVICES,
    build_model_status,
    write_model_status_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the TIAToolbox pretrained BCSS model as a reproducible smoke test. "
            "This script does not run inference, download datasets, or train models."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="TIAToolbox pretrained model name to load.",
    )
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="auto",
        help="Device selection: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_STATUS_JSON,
        help="Path where the model-load status JSON will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("TIAToolbox model smoke test")
    print("===========================")
    print(f"Model name: {args.model_name}")
    print(f"Requested device: {args.device}")
    print("This script only validates model loading; it does not run inference.")
    print("Loading pretrained model...")

    status = build_model_status(
        model_name=args.model_name,
        requested_device=args.device,
        root_dir=ROOT_DIR,
    )
    output_path = write_model_status_json(status, args.output_json)

    print(f"TIAToolbox version: {status.get('tiatoolbox_version')}")
    print(f"Torch version: {status.get('torch_version')}")
    print(f"Resolved device: {status.get('resolved_device')}")

    if status["status"] == "loaded":
        print("Model loaded: OK")
        print(f"Model class: {status.get('model_class')}")
        if status.get("ioconfig_class"):
            print(f"IO config class: {status.get('ioconfig_class')}")
    else:
        print("Model loaded: FAILED")
        print(f"Reason: {status.get('error')}")
        print(f"Suggested next step: {status.get('suggested_next_step')}")

    for warning in status.get("warnings", []):
        print(f"Warning: {warning}")

    print(f"Status JSON: {output_path}")
    return 0 if status["status"] == "loaded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
