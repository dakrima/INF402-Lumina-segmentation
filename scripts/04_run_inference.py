#!/usr/bin/env python
"""Run a controlled TIAToolbox inference smoke test on a small local image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.tiatoolbox_inference import run_inference_smoke_test  # noqa: E402
from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, SUPPORTED_DEVICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small technical inference smoke test with TIAToolbox. "
            "This does not evaluate quality, diagnose, calculate RCB, or train models."
        ),
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="auto",
        help="Device selection: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference_smoke/test_reconstructed"),
    )
    parser.add_argument(
        "--input-mode",
        choices=["patch", "wsi"],
        default="patch",
        help="Use patch for small tiles without WSI metadata, or wsi for whole-slide inputs.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Safely clear the output directory before running inference.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Opacity for the technical prediction overlay.",
    )
    parser.add_argument(
        "--strict-input-validation",
        action="store_true",
        help=(
            "Fail before inference if technical input validation detects an "
            "incompatible patch shape, dtype, range, or channel count."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("TIAToolbox inference smoke test")
    print("===============================")
    print(f"Image: {args.image_path}")
    print(f"Model name: {args.model_name}")
    print(f"Requested device: {args.device}")
    print(f"Input mode: {args.input_mode}")
    print(f"Output dir: {args.output_dir}")
    print(f"Strict input validation: {args.strict_input_validation}")
    print("Clinical warning: technical smoke test only; not diagnosis, not RCB.")

    summary, summary_path = run_inference_smoke_test(
        image_path=args.image_path,
        output_dir=args.output_dir,
        root_dir=ROOT_DIR,
        model_name=args.model_name,
        requested_device=args.device,
        input_mode=args.input_mode,
        overlay_alpha=args.overlay_alpha,
        clear_output=args.clear_output,
        strict_input_validation=args.strict_input_validation,
    )

    print(f"Status: {summary['status']}")
    print(f"Patch mode: {summary.get('patch_mode')}")
    print(f"Resolved device: {summary.get('resolved_device')}")
    print(f"TIAToolbox version: {summary.get('tiatoolbox_version')}")
    print(f"Torch version: {summary.get('torch_version')}")
    print(f"Prediction shape: {summary.get('prediction_shape')}")
    print(f"Raw prediction shape: {summary.get('raw_prediction_shape')}")
    print(f"Visualized mask shape: {summary.get('visualized_mask_shape')}")
    print(f"Resized for visualization: {summary.get('resized_for_visualization')}")
    print(f"Unique prediction values: {summary.get('unique_prediction_values')}")
    print(f"Input validation: {summary.get('input_validation', {}).get('status')}")
    print(f"Class mapping source: {summary.get('class_mapping_source')}")
    print(f"Legend JSON: {summary.get('legend_json')}")
    print(f"Legend PNG: {summary.get('legend_png')}")
    print(f"Summary JSON: {summary_path}")

    if summary["status"] == "completed":
        print(f"Input preview: {summary['outputs']['input_preview']}")
        print(f"Prediction mask: {summary['outputs']['prediction_mask']}")
        print(f"Prediction overlay: {summary['outputs']['prediction_overlay']}")
        print(f"Overlay with legend: {summary['outputs']['prediction_overlay_with_legend']}")
        for warning in summary.get("warnings", []):
            print(f"Warning: {warning}")
        return 0

    print(f"Error: {summary.get('error')}")
    print(f"Suggested next step: {summary.get('suggested_next_step')}")
    for warning in summary.get("warnings", []):
        print(f"Warning: {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
