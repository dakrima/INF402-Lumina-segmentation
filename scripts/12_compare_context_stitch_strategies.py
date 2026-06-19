#!/usr/bin/env python
"""Compare no-overlap and overlap-aware context stitching on selected patches."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.context_stitch_comparison import (  # noqa: E402
    CLINICAL_WARNING,
    StrategyComparisonConfig,
    compare_context_stitch_strategies,
)
from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, SUPPORTED_DEVICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare technical context stitching strategies over selected WSI patches. "
            "This is technical segmentation/inference only; it does not diagnose, "
            "calculate RCB, or validate clinical performance."
        ),
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs"),
        help="Selection run directory with selected_metadata.csv and selection_summary.json.",
    )
    parser.add_argument("--patch-index", type=int, default=0, help="Single selected patch index.")
    parser.add_argument(
        "--patch-indices",
        default=None,
        help="Comma-separated selected patch indices, e.g. 0,1,2. Maximum 3.",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=None,
        help="Process the first N selected patches. Maximum 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/context_stitch_comparison"),
        help="Output directory for comparison artifacts.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="cpu",
        help="Device for TIAToolbox inference.",
    )
    parser.add_argument(
        "--overlap-stride",
        type=int,
        default=450,
        help="Stride used by overlap-aware output coordinates.",
    )
    parser.add_argument(
        "--blend-mode",
        choices=["uniform", "feathered", "both"],
        default="uniform",
        help="Probability blending mode for overlap-aware reconstruction.",
    )
    parser.add_argument(
        "--run-no-overlap",
        action="store_true",
        default=True,
        help="Run context-stitch-2x2 without output overlap. Enabled by default.",
    )
    parser.add_argument(
        "--run-overlap-aware",
        action="store_true",
        default=True,
        help="Run overlap-aware probability blending. Enabled by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output directory. Only safe repo output paths are cleared.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (ROOT_DIR / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = ROOT_DIR.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        resolved_root,
        resolved_root / "data",
        resolved_root / "outputs",
    }
    if resolved_output.exists() and any(resolved_output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {resolved_output}. "
            "Use --overwrite to regenerate comparison outputs."
        )
    if overwrite and resolved_output.exists():
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite only clears output directories inside the repository.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def _parse_patch_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if args.patch_indices:
        indices = tuple(
            int(item.strip()) for item in str(args.patch_indices).split(",") if item.strip()
        )
    elif args.num_patches is not None:
        if args.num_patches < 1:
            raise ValueError("--num-patches must be positive.")
        indices = tuple(range(args.num_patches))
    else:
        indices = (args.patch_index,)
    if not indices:
        raise ValueError("At least one patch index is required.")
    if len(indices) > 3:
        raise ValueError("Initial context-stitch comparison is capped at 3 patches.")
    if any(index < 0 for index in indices):
        raise ValueError("Patch indices must be non-negative.")
    return indices


def main() -> int:
    args = parse_args()
    try:
        selection_dir = _resolve_path(args.selection_dir)
        output_dir = _resolve_path(args.output_dir)
        patch_indices = _parse_patch_indices(args)
        if not selection_dir.exists():
            raise FileNotFoundError(f"Selection directory does not exist: {selection_dir}")
        _prepare_output_dir(output_dir=output_dir, overwrite=args.overwrite)

        print("Context-stitch strategy comparison")
        print("==================================")
        print(f"Selection dir: {selection_dir}")
        print(f"Patch indices: {patch_indices}")
        print(f"Output dir: {output_dir}")
        print(f"Model name: {args.model_name}")
        print(f"Requested device: {args.device}")
        print(f"Overlap stride: {args.overlap_stride}")
        print(f"Blend mode: {args.blend_mode}")
        print(f"Clinical warning: {CLINICAL_WARNING}")

        summary = compare_context_stitch_strategies(
            StrategyComparisonConfig(
                selection_dir=selection_dir,
                output_dir=output_dir,
                patch_indices=patch_indices,
                model_name=args.model_name,
                device=args.device,
                overlap_stride=args.overlap_stride,
                blend_mode=args.blend_mode,
                run_no_overlap=args.run_no_overlap,
                run_overlap_aware=args.run_overlap_aware,
            )
        )
        print(f"[OK] Status: {summary.get('status')}")
        print(f"[OK] Recommendation: {summary.get('recommendation')}")
        print(f"[OK] Patches processed: {summary.get('num_patches_processed')}")
        print(f"[OK] Failures: {summary.get('num_failures')}")
        print(f"[OK] Summary JSON: {summary.get('outputs', {}).get('strategy_comparison_summary_json')}")
        print(f"[OK] Metrics CSV: {summary.get('outputs', {}).get('strategy_comparison_metrics_csv')}")
        return 0 if summary.get("status") == "completed" else 1
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
