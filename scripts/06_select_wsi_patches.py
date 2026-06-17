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
SMART_V2_LIGHT_SELECTOR_NAME = "smart_tissue_nuclei_v2_light"
V3_SERVER_QUALITY_SELECTOR_NAME = "v3_server_quality"
SUPPORTED_SELECTORS = (
    BASELINE_SELECTOR_NAME,
    SMART_SELECTOR_NAME,
    SMART_V2_LIGHT_SELECTOR_NAME,
    V3_SERVER_QUALITY_SELECTOR_NAME,
)


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
        default=None,
        help=(
            "For smart/v3 selectors. 0 scores all thumbnail-filtered candidates; "
            "N > 0 scores at most N candidates after seeded shuffle. Defaults to "
            "300 for smart v1/v2 and 2000 for v3_server_quality."
        ),
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=None,
        help=(
            "For smart/v3 selectors. Downsampled patch size used for features. "
            "Defaults to 256 for smart v1/v2 and 512 for v3_server_quality."
        ),
    )
    parser.add_argument(
        "--lambda-spatial",
        type=float,
        default=0.15,
        help="For smart selectors. Spatial redundancy penalty weight.",
    )
    parser.add_argument(
        "--min-distance-level0",
        type=int,
        default=None,
        help=(
            "For smart selectors. Distance scale for spatial penalty. "
            "Defaults to --patch-size."
        ),
    )
    parser.add_argument(
        "--nuclear-proxy",
        choices=("rgb_purple", "hed_deconvolution"),
        default=None,
        help="For smart selectors. Defaults to rgb_purple for v1 and hed_deconvolution for v2_light.",
    )
    parser.add_argument(
        "--spatial-strategy",
        choices=("penalty", "quotas"),
        default=None,
        help="For smart selectors. Defaults to penalty for v1 and quotas for v2_light.",
    )
    parser.add_argument(
        "--quota-grid",
        default="4x4",
        help="For spatial quotas. Format ROWSxCOLS, for example 4x4.",
    )
    parser.add_argument(
        "--quota-min-score-quantile",
        type=float,
        default=None,
        help=(
            "Minimum score_raw quantile used by soft quota selection. Defaults to "
            "0.25 for smart v2_light and 0.20 for v3_server_quality."
        ),
    )
    parser.add_argument(
        "--diversity-strategy",
        choices=("none", "farthest_feature"),
        default=None,
        help="For smart selectors. Defaults to none for v1 and farthest_feature for v2_light.",
    )
    parser.add_argument(
        "--feature-diversity-weight",
        type=float,
        default=None,
        help=(
            "Weight for farthest_feature diversity bonus. Defaults to 0.10 for "
            "smart v2_light and 0.15 for v3_server_quality."
        ),
    )
    parser.add_argument(
        "--redundancy-penalty-weight",
        type=float,
        default=0.10,
        help="For v3_server_quality. Feature-space redundancy penalty weight.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.15,
        help="For v3_server_quality. Minimum technical quality score for preferred selection.",
    )
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="For v3_server_quality. Write scored_candidates.csv feature cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "For v3_server_quality. Reuse a compatible scored_candidates.csv cache "
            "when present; combine with --overwrite to rebuild outputs from cache."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("debug", "minimal", "full"),
        default="debug",
        help="For v3_server_quality. Controls optional debug artifacts.",
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
            SMART_V2_LIGHT_SELECTOR_NAME as PACKAGE_SMART_V2_LIGHT_SELECTOR_NAME,
            SmartTissueNucleiConfig,
            V3_SERVER_QUALITY_SELECTOR_NAME as PACKAGE_V3_SERVER_QUALITY_SELECTOR_NAME,
            V3ServerQualityConfig,
            run_baseline_selection,
            run_smart_tissue_nuclei_selection,
            run_v3_server_quality_selection,
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
    elif args.selector in (SMART_SELECTOR_NAME, SMART_V2_LIGHT_SELECTOR_NAME):
        is_v2_light = args.selector == SMART_V2_LIGHT_SELECTOR_NAME
        nuclear_proxy = args.nuclear_proxy or (
            "hed_deconvolution" if is_v2_light else "rgb_purple"
        )
        spatial_strategy = args.spatial_strategy or ("quotas" if is_v2_light else "penalty")
        diversity_strategy = args.diversity_strategy or (
            "farthest_feature" if is_v2_light else "none"
        )
        max_candidates_to_score = (
            args.max_candidates_to_score
            if args.max_candidates_to_score is not None
            else 300
        )
        feature_size = args.feature_size if args.feature_size is not None else 256
        quota_min_score_quantile = (
            args.quota_min_score_quantile
            if args.quota_min_score_quantile is not None
            else 0.25
        )
        feature_diversity_weight = (
            args.feature_diversity_weight
            if args.feature_diversity_weight is not None
            else 0.10
        )
        if is_v2_light and PACKAGE_SMART_V2_LIGHT_SELECTOR_NAME != SMART_V2_LIGHT_SELECTOR_NAME:
            print("[FAIL] Internal selector constant mismatch for smart_tissue_nuclei_v2_light.")
            return 1
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
            max_candidates_to_score=max_candidates_to_score,
            feature_size=feature_size,
            lambda_spatial=args.lambda_spatial,
            min_distance_level0=args.min_distance_level0,
            nuclear_proxy=nuclear_proxy,
            spatial_strategy=spatial_strategy,
            quota_grid=args.quota_grid,
            quota_min_score_quantile=quota_min_score_quantile,
            diversity_strategy=diversity_strategy,
            feature_diversity_weight=feature_diversity_weight,
        )
        runner = run_smart_tissue_nuclei_selection
    else:
        if PACKAGE_V3_SERVER_QUALITY_SELECTOR_NAME != V3_SERVER_QUALITY_SELECTOR_NAME:
            print("[FAIL] Internal selector constant mismatch for v3_server_quality.")
            return 1
        max_candidates_to_score = (
            args.max_candidates_to_score
            if args.max_candidates_to_score is not None
            else 2000
        )
        feature_size = args.feature_size if args.feature_size is not None else 512
        quota_min_score_quantile = (
            args.quota_min_score_quantile
            if args.quota_min_score_quantile is not None
            else 0.20
        )
        feature_diversity_weight = (
            args.feature_diversity_weight
            if args.feature_diversity_weight is not None
            else 0.15
        )
        config = V3ServerQualityConfig(
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
            max_candidates_to_score=max_candidates_to_score,
            feature_size=feature_size,
            lambda_spatial=args.lambda_spatial,
            min_distance_level0=args.min_distance_level0,
            quota_grid=args.quota_grid,
            quota_min_score_quantile=quota_min_score_quantile,
            feature_diversity_weight=feature_diversity_weight,
            redundancy_penalty_weight=args.redundancy_penalty_weight,
            min_quality_score=args.min_quality_score,
            resume=args.resume,
            cache_features=args.cache_features,
            output_mode=args.output_mode,
        )
        runner = run_v3_server_quality_selection

    print("WSI patch selection")
    print("===================")
    print(f"Selector: {args.selector}")
    print(f"WSI path: {args.wsi_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Patch size / stride: {args.patch_size} / {args.stride}")
    print(f"Max patches: {args.max_patches}")
    print(f"Minimum tissue ratio: {args.min_tissue_ratio}")
    print(f"Seed: {args.seed}")
    if args.selector in (SMART_SELECTOR_NAME, SMART_V2_LIGHT_SELECTOR_NAME, V3_SERVER_QUALITY_SELECTOR_NAME):
        print(f"Max candidates to score: {config.max_candidates_to_score}")
        print(f"Feature size: {config.feature_size}")
        print(f"Lambda spatial: {args.lambda_spatial}")
        print(f"Minimum distance level 0: {args.min_distance_level0 or args.patch_size}")
        print(f"Nuclear proxy: {getattr(config, 'nuclear_proxy', 'hed_deconvolution')}")
        print(f"Spatial strategy: {getattr(config, 'spatial_strategy', 'quotas')}")
        print(f"Quota grid: {config.quota_grid}")
        print(f"Diversity strategy: {getattr(config, 'diversity_strategy', 'farthest_feature')}")
        if args.selector == V3_SERVER_QUALITY_SELECTOR_NAME:
            print(f"Minimum quality score: {config.min_quality_score}")
            print(f"Redundancy penalty weight: {config.redundancy_penalty_weight}")
            print(f"Cache features: {config.cache_features}")
            print(f"Resume: {config.resume}")
            print(f"Output mode: {config.output_mode}")
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
