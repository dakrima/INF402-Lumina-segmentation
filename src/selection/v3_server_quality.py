"""Server-oriented non-DL patch selector for technical downstream segmentation.

This selector is designed for INF402/Lumina server execution. It does not use
deep learning, does not use the segmentation model for selection, and does not
make clinical claims. It ranks patches with technical proxies intended to
prioritize segmentable, traceable, non-redundant regions for later semantic
segmentation.
"""

from __future__ import annotations

import csv
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.wsi_patch_extraction import (
    SUPPORTED_WSI_EXTENSIONS,
    TISSUE_MASK_METHOD,
    _import_openslide,
)
from src.selection.candidate_generation import PatchCandidate, generate_tissue_candidates
from src.selection.diversity import (
    assign_spatial_regions,
    feature_diversity_bonus,
    proximity_penalty,
    score_quantile,
)
from src.selection.manifests import (
    CANDIDATE_METADATA_FIELDS,
    SELECTED_METADATA_FIELDS,
    utc_now_iso,
    write_csv_manifest,
    write_json_manifest,
)
from src.selection.previews import save_wsi_patch_selection_preview
from src.selection.quality_filters import (
    ARTIFACT_PENALTY_METHOD,
    BLUR_SCORE_METHOD,
    HED_NUCLEAR_SIGNAL_METHOD,
    RGB_NUCLEAR_SIGNAL_METHOD,
    VISUAL_ENTROPY_METHOD,
    compute_patch_features,
)
from src.selection.scoring import normalize_feature
from src.selection.tiatoolbox_baseline import (
    CANDIDATE_METADATA_SEMANTICS,
    CANDIDATE_POOL,
    CLINICAL_WARNING,
    _base_slide_metadata,
    _prepare_output_dir,
    _resolve_output_dir,
)


V3_SERVER_QUALITY_SELECTOR_NAME = "v3_server_quality"
V3_SELECTOR_VERSION = "v3_server_quality_1.0"
V3_CANDIDATE_ORDERING = "thumbnail_filtered_seeded_shuffle_then_server_quality_score"
V3_PREVIEW_SHOWS = "scored_candidates_with_selected_highlighted"
V3_FEATURE_CACHE_FILE = "scored_candidates.csv"
USEFUL_PATCH_DEFINITION = (
    "A useful patch is technically segmentable, spatially traceable, and "
    "non-redundant, with expected utility for estimating downstream predicted "
    "class distributions in the conceptual post-neoadjuvant tumor bed target, "
    "including both higher-cellularity informative regions and lower-cellularity "
    "treated-bed-like tissue proxies."
)
NO_MODEL_SELECTION_NOTE = (
    "v3_server_quality uses technical image proxies only. No deep learning model "
    "and no segmentation model are used for patch selection."
)

V3_WEIGHTS = {
    "technical_quality_score": 0.30,
    "residual_candidate_proxy": 0.22,
    "low_cellularity_treated_bed_proxy": 0.18,
    "tumor_bed_relevance_proxy": 0.20,
    "heterogeneity_score": 0.10,
    "artifact_penalty_norm": -0.10,
}

V3_FEATURE_DIVERSITY_FIELDS = [
    "technical_quality_score",
    "tumor_bed_relevance_proxy",
    "residual_candidate_proxy",
    "low_cellularity_treated_bed_proxy",
    "heterogeneity_score",
    "cellularity_proxy_score",
    "tissue_ratio_norm",
    "thumbnail_tissue_ratio_norm",
    "x_norm",
    "y_norm",
]

V3_NORMALIZED_FIELDS = [
    "tissue_ratio_norm",
    "nuclear_signal_rgb_norm",
    "nuclear_signal_hed_norm",
    "visual_entropy_norm",
    "blur_score_norm",
    "artifact_penalty_norm",
    "thumbnail_tissue_ratio_norm",
    "x_norm",
    "y_norm",
]

V3_CRITICAL_NUMERIC_FIELDS = [
    "technical_quality_score",
    "heterogeneity_score",
    "cellularity_proxy_score",
    "residual_candidate_proxy",
    "low_cellularity_treated_bed_proxy",
    "tumor_bed_relevance_proxy",
    "usefulness_score",
    "redundancy_penalty",
    "score_final",
]

SCORED_CANDIDATE_FIELDS = list(
    dict.fromkeys(
        [
            *CANDIDATE_METADATA_FIELDS,
            "feature_size",
            *V3_NORMALIZED_FIELDS,
        ]
    )
)


@dataclass(frozen=True)
class V3ServerQualityConfig:
    """Configuration for the server-oriented non-DL v3 patch selector."""

    wsi_path: Path
    output_dir: Path
    root_dir: Path
    selector: str = V3_SERVER_QUALITY_SELECTOR_NAME
    patch_size: int = 1024
    stride: int = 1024
    max_patches: int = 16
    min_tissue_ratio: float = 0.20
    seed: int = 42
    thumbnail_max_size: int = 2048
    overwrite: bool = False
    max_candidates_to_score: int = 2000
    feature_size: int = 512
    lambda_spatial: float = 0.15
    min_distance_level0: int | None = None
    quota_grid: str = "4x4"
    quota_min_score_quantile: float = 0.20
    feature_diversity_weight: float = 0.15
    redundancy_penalty_weight: float = 0.10
    min_quality_score: float = 0.15
    resume: bool = False
    cache_features: bool = False
    output_mode: str = "debug"


def _validate_config(config: V3ServerQualityConfig, wsi_path: Path) -> None:
    if config.selector != V3_SERVER_QUALITY_SELECTOR_NAME:
        raise NotImplementedError(
            f"Selector '{config.selector}' is not handled by v3_server_quality."
        )
    if config.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if config.stride <= 0:
        raise ValueError("--stride must be positive.")
    if config.max_patches <= 0:
        raise ValueError("--max-patches must be positive.")
    if not 0 <= config.min_tissue_ratio <= 1:
        raise ValueError("--min-tissue-ratio must be between 0 and 1.")
    if config.thumbnail_max_size <= 0:
        raise ValueError("--thumbnail-max-size must be positive.")
    if config.max_candidates_to_score < 0:
        raise ValueError("--max-candidates-to-score must be >= 0.")
    if config.feature_size <= 0:
        raise ValueError("--feature-size must be positive.")
    if config.lambda_spatial < 0:
        raise ValueError("--lambda-spatial must be >= 0.")
    if config.min_distance_level0 is not None and config.min_distance_level0 <= 0:
        raise ValueError("--min-distance-level0 must be positive when provided.")
    if not 0 <= config.quota_min_score_quantile <= 1:
        raise ValueError("--quota-min-score-quantile must be between 0 and 1.")
    if config.feature_diversity_weight < 0:
        raise ValueError("--feature-diversity-weight must be >= 0.")
    if config.redundancy_penalty_weight < 0:
        raise ValueError("--redundancy-penalty-weight must be >= 0.")
    if not 0 <= config.min_quality_score <= 1:
        raise ValueError("--min-quality-score must be between 0 and 1.")
    if config.output_mode not in {"debug", "minimal", "full"}:
        raise ValueError("--output-mode must be debug, minimal, or full.")
    if wsi_path.suffix.lower() not in SUPPORTED_WSI_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_WSI_EXTENSIONS))
        raise ValueError(f"Unsupported WSI extension '{wsi_path.suffix}'. Use one of: {allowed}.")
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")


def _format_float(value: object) -> str:
    return f"{float(value):.6f}"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _candidate_pool_row(
    candidate: PatchCandidate,
    *,
    config: V3ServerQualityConfig,
    wsi_path: Path,
    slide_metadata: dict[str, Any],
) -> dict[str, object]:
    row = {
        "candidate_id": candidate.candidate_id,
        "grid_index": candidate.grid_index,
        "x_level0": candidate.x_level0,
        "y_level0": candidate.y_level0,
        "patch_size": candidate.patch_size,
        "width": "",
        "height": "",
        "thumbnail_tissue_ratio": _format_float(candidate.thumbnail_tissue_ratio),
        "evaluated": False,
        "scored": False,
        "nuclear_proxy": "hed_deconvolution",
        "tissue_ratio": "",
        "nuclear_signal": "",
        "nuclear_signal_rgb": "",
        "nuclear_signal_hed": "",
        "visual_entropy": "",
        "blur_score": "",
        "artifact_penalty": "",
        "technical_quality_score": "",
        "heterogeneity_score": "",
        "cellularity_proxy_score": "",
        "residual_candidate_proxy": "",
        "low_cellularity_treated_bed_proxy": "",
        "tumor_bed_relevance_proxy": "",
        "usefulness_score": "",
        "spatial_penalty": "",
        "feature_diversity_bonus": "",
        "redundancy_penalty": "",
        "score_raw": "",
        "score_final": "",
        "usefulness_reason": "",
        "region_id": "",
        "region_row": "",
        "region_col": "",
        "quota_grid": config.quota_grid,
        "spatial_strategy": "quotas",
        "diversity_strategy": "farthest_feature",
        "selected": False,
        "rank": "",
        "filename": "",
        "patch_id": "",
        "patch_path": "",
        "selector": config.selector,
        "selection_method": config.selector,
        "seed": config.seed,
        "source_wsi_path": str(wsi_path),
        **slide_metadata,
    }
    return row


def _selected_row(candidate_row: dict[str, object]) -> dict[str, object]:
    return {
        "patch_id": candidate_row["patch_id"],
        "filename": candidate_row["filename"],
        "selected": True,
        "rank": candidate_row["rank"],
        "x_level0": candidate_row["x_level0"],
        "y_level0": candidate_row["y_level0"],
        "patch_size": candidate_row["patch_size"],
        "width": candidate_row["width"],
        "height": candidate_row["height"],
        "thumbnail_tissue_ratio": candidate_row["thumbnail_tissue_ratio"],
        "tissue_ratio": candidate_row["tissue_ratio"],
        "nuclear_signal": candidate_row["nuclear_signal"],
        "nuclear_signal_rgb": candidate_row["nuclear_signal_rgb"],
        "nuclear_signal_hed": candidate_row["nuclear_signal_hed"],
        "visual_entropy": candidate_row["visual_entropy"],
        "blur_score": candidate_row["blur_score"],
        "artifact_penalty": candidate_row["artifact_penalty"],
        "technical_quality_score": candidate_row["technical_quality_score"],
        "heterogeneity_score": candidate_row["heterogeneity_score"],
        "cellularity_proxy_score": candidate_row["cellularity_proxy_score"],
        "residual_candidate_proxy": candidate_row["residual_candidate_proxy"],
        "low_cellularity_treated_bed_proxy": candidate_row["low_cellularity_treated_bed_proxy"],
        "tumor_bed_relevance_proxy": candidate_row["tumor_bed_relevance_proxy"],
        "usefulness_score": candidate_row["usefulness_score"],
        "spatial_penalty": candidate_row["spatial_penalty"],
        "feature_diversity_bonus": candidate_row["feature_diversity_bonus"],
        "redundancy_penalty": candidate_row["redundancy_penalty"],
        "score_raw": candidate_row["score_raw"],
        "score_final": candidate_row["score_final"],
        "usefulness_reason": candidate_row["usefulness_reason"],
        "nuclear_proxy": candidate_row["nuclear_proxy"],
        "region_id": candidate_row["region_id"],
        "region_row": candidate_row["region_row"],
        "region_col": candidate_row["region_col"],
        "quota_grid": candidate_row["quota_grid"],
        "spatial_strategy": candidate_row["spatial_strategy"],
        "diversity_strategy": candidate_row["diversity_strategy"],
        "patch_path": candidate_row["patch_path"],
        "selector": candidate_row["selector"],
        "source_wsi_path": candidate_row["source_wsi_path"],
        "slide_width": candidate_row["slide_width"],
        "slide_height": candidate_row["slide_height"],
        "objective_power": candidate_row["objective_power"],
        "mpp_x": candidate_row["mpp_x"],
        "mpp_y": candidate_row["mpp_y"],
        "level_count": candidate_row["level_count"],
        "selection_method": candidate_row["selection_method"],
        "seed": candidate_row["seed"],
    }


def _method_config(config: V3ServerQualityConfig, *, min_distance_level0: int) -> dict[str, object]:
    return {
        "selector": config.selector,
        "selector_name": config.selector,
        "version": V3_SELECTOR_VERSION,
        "candidate_pool": CANDIDATE_POOL,
        "candidate_metadata_semantics": CANDIDATE_METADATA_SEMANTICS,
        "candidate_ordering": V3_CANDIDATE_ORDERING,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "seed": config.seed,
        "thumbnail_max_size": config.thumbnail_max_size,
        "feature_size": config.feature_size,
        "max_candidates_to_score": config.max_candidates_to_score,
        "weights": V3_WEIGHTS,
        "lambda_spatial": config.lambda_spatial,
        "min_distance_level0": min_distance_level0,
        "quota_grid": config.quota_grid,
        "quota_min_score_quantile": config.quota_min_score_quantile,
        "feature_diversity_weight": config.feature_diversity_weight,
        "redundancy_penalty_weight": config.redundancy_penalty_weight,
        "min_quality_score": config.min_quality_score,
        "resume": config.resume,
        "cache_features": config.cache_features,
        "output_mode": config.output_mode,
        "useful_patch_definition": USEFUL_PATCH_DEFINITION,
        "no_deep_learning_used_for_selection": True,
        "segmentation_model_used_for_selection": False,
        "selection_model_note": NO_MODEL_SELECTION_NOTE,
        "tissue_mask_method": TISSUE_MASK_METHOD,
        "nuclear_signal_rgb_method": RGB_NUCLEAR_SIGNAL_METHOD,
        "nuclear_signal_hed_method": HED_NUCLEAR_SIGNAL_METHOD,
        "visual_entropy_method": VISUAL_ENTROPY_METHOD,
        "blur_score_method": BLUR_SCORE_METHOD,
        "artifact_penalty_method": ARTIFACT_PENALTY_METHOD,
        "clinical_warning": CLINICAL_WARNING,
        "created_at": utc_now_iso(),
    }


def _select_candidates_to_score(
    candidates: list[PatchCandidate],
    *,
    seed: int,
    max_candidates_to_score: int,
) -> list[PatchCandidate]:
    ordered_candidates = list(candidates)
    random.Random(seed).shuffle(ordered_candidates)
    if max_candidates_to_score == 0:
        return ordered_candidates
    return ordered_candidates[:max_candidates_to_score]


def _compute_v3_raw_features(
    patch_image: object,
    *,
    feature_size: int,
) -> dict[str, float]:
    hed_features = compute_patch_features(
        rgb_patch=patch_image,
        feature_size=feature_size,
        nuclear_proxy="hed_deconvolution",
    )
    rgb_features = compute_patch_features(
        rgb_patch=patch_image,
        feature_size=feature_size,
        nuclear_proxy="rgb_purple",
    )
    return {
        "tissue_ratio": float(hed_features["tissue_ratio"]),
        "nuclear_signal": float(hed_features["nuclear_signal"]),
        "nuclear_signal_hed": float(hed_features["nuclear_signal"]),
        "nuclear_signal_rgb": float(rgb_features["nuclear_signal"]),
        "visual_entropy": float(hed_features["visual_entropy"]),
        "blur_score": float(hed_features["blur_score"]),
        "artifact_penalty": float(hed_features["artifact_penalty"]),
    }


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _usefulness_reason(record: dict[str, object]) -> str:
    residual = _safe_float(record.get("residual_candidate_proxy"))
    treated = _safe_float(record.get("low_cellularity_treated_bed_proxy"))
    heterogeneity = _safe_float(record.get("heterogeneity_score"))
    quality = _safe_float(record.get("technical_quality_score"))

    if treated >= max(residual, heterogeneity, quality):
        return "low_cellularity_treated_bed_proxy"
    if residual >= max(treated, heterogeneity, quality):
        return "higher_cellularity_residual_candidate_proxy"
    if heterogeneity >= max(treated, residual, quality):
        return "heterogeneous_region_proxy"
    return "technical_quality_balanced_proxy"


def _apply_v3_scores(records: list[dict[str, object]]) -> None:
    normalized_by_feature = {
        "tissue_ratio": normalize_feature(record.get("tissue_ratio") for record in records),
        "nuclear_signal_rgb": normalize_feature(
            record.get("nuclear_signal_rgb") for record in records
        ),
        "nuclear_signal_hed": normalize_feature(
            record.get("nuclear_signal_hed") for record in records
        ),
        "visual_entropy": normalize_feature(
            record.get("visual_entropy") for record in records
        ),
        "blur_score": normalize_feature(record.get("blur_score") for record in records),
        "artifact_penalty": normalize_feature(
            record.get("artifact_penalty") for record in records
        ),
        "thumbnail_tissue_ratio": normalize_feature(
            record.get("thumbnail_tissue_ratio") for record in records
        ),
        "x_level0": normalize_feature(record.get("x_level0") for record in records),
        "y_level0": normalize_feature(record.get("y_level0") for record in records),
    }

    for index, record in enumerate(records):
        tissue_norm = normalized_by_feature["tissue_ratio"][index]
        rgb_norm = normalized_by_feature["nuclear_signal_rgb"][index]
        hed_norm = normalized_by_feature["nuclear_signal_hed"][index]
        entropy_norm = normalized_by_feature["visual_entropy"][index]
        blur_norm = normalized_by_feature["blur_score"][index]
        artifact_norm = normalized_by_feature["artifact_penalty"][index]
        thumbnail_norm = normalized_by_feature["thumbnail_tissue_ratio"][index]
        x_norm = normalized_by_feature["x_level0"][index]
        y_norm = normalized_by_feature["y_level0"][index]

        artifact_quality = 1.0 - artifact_norm
        cellularity_proxy = _clip01(0.70 * hed_norm + 0.30 * rgb_norm)
        technical_quality = _clip01(
            0.42 * tissue_norm
            + 0.28 * blur_norm
            + 0.20 * artifact_quality
            + 0.10 * thumbnail_norm
        )
        heterogeneity = _clip01(entropy_norm)
        residual_candidate = _clip01(
            0.45 * cellularity_proxy
            + 0.30 * heterogeneity
            + 0.25 * technical_quality
        )
        low_cellularity_treated_bed = _clip01(
            technical_quality
            * (0.55 * tissue_norm + 0.25 * heterogeneity + 0.20 * artifact_quality)
            * (1.0 - 0.75 * cellularity_proxy)
        )
        tumor_bed_relevance = _clip01(
            0.32 * technical_quality
            + 0.28 * residual_candidate
            + 0.25 * low_cellularity_treated_bed
            + 0.15 * heterogeneity
        )
        usefulness = _clip01(
            V3_WEIGHTS["technical_quality_score"] * technical_quality
            + V3_WEIGHTS["residual_candidate_proxy"] * residual_candidate
            + V3_WEIGHTS["low_cellularity_treated_bed_proxy"] * low_cellularity_treated_bed
            + V3_WEIGHTS["tumor_bed_relevance_proxy"] * tumor_bed_relevance
            + V3_WEIGHTS["heterogeneity_score"] * heterogeneity
            + V3_WEIGHTS["artifact_penalty_norm"] * artifact_norm
        )

        record["tissue_ratio_norm"] = tissue_norm
        record["nuclear_signal_rgb_norm"] = rgb_norm
        record["nuclear_signal_hed_norm"] = hed_norm
        record["visual_entropy_norm"] = entropy_norm
        record["blur_score_norm"] = blur_norm
        record["artifact_penalty_norm"] = artifact_norm
        record["thumbnail_tissue_ratio_norm"] = thumbnail_norm
        record["x_norm"] = x_norm
        record["y_norm"] = y_norm
        record["technical_quality_score"] = technical_quality
        record["heterogeneity_score"] = heterogeneity
        record["cellularity_proxy_score"] = cellularity_proxy
        record["residual_candidate_proxy"] = residual_candidate
        record["low_cellularity_treated_bed_proxy"] = low_cellularity_treated_bed
        record["tumor_bed_relevance_proxy"] = tumor_bed_relevance
        record["usefulness_score"] = usefulness
        record["score_raw"] = usefulness
        record["usefulness_reason"] = _usefulness_reason(record)


def _selection_score(
    record: dict[str, object],
    selected_records: list[dict[str, object]],
    *,
    lambda_spatial: float,
    min_distance_level0: float,
    feature_diversity_weight: float,
    redundancy_penalty_weight: float,
) -> tuple[float, float, float, float]:
    spatial_penalty = proximity_penalty(
        candidate=record,
        selected_records=selected_records,
        min_distance_level0=min_distance_level0,
    )
    diversity_bonus = (
        feature_diversity_bonus(
            candidate=record,
            selected_records=selected_records,
            feature_names=V3_FEATURE_DIVERSITY_FIELDS,
        )
        if selected_records
        else 0.0
    )
    redundancy_penalty = (1.0 - min(1.0, diversity_bonus)) if selected_records else 0.0
    score_final = (
        _safe_float(record.get("score_raw"))
        - lambda_spatial * spatial_penalty
        - redundancy_penalty_weight * redundancy_penalty
        + feature_diversity_weight * diversity_bonus
    )
    return (
        float(score_final),
        float(spatial_penalty),
        float(diversity_bonus),
        float(redundancy_penalty),
    )


def _pick_best_record(
    available: list[dict[str, object]],
    selected: list[dict[str, object]],
    *,
    lambda_spatial: float,
    min_distance_level0: float,
    feature_diversity_weight: float,
    redundancy_penalty_weight: float,
) -> tuple[dict[str, object] | None, float, float, float, float]:
    best_record: dict[str, object] | None = None
    best_score = -math.inf
    best_spatial_penalty = 0.0
    best_diversity_bonus = 0.0
    best_redundancy_penalty = 0.0

    for record in available:
        score, spatial_penalty, diversity_bonus, redundancy_penalty = _selection_score(
            record,
            selected,
            lambda_spatial=lambda_spatial,
            min_distance_level0=min_distance_level0,
            feature_diversity_weight=feature_diversity_weight,
            redundancy_penalty_weight=redundancy_penalty_weight,
        )
        tie_breaker = str(record.get("candidate_id", ""))
        best_tie_breaker = str(best_record.get("candidate_id", "")) if best_record else ""
        if score > best_score or (
            math.isclose(score, best_score) and tie_breaker < best_tie_breaker
        ):
            best_record = record
            best_score = score
            best_spatial_penalty = spatial_penalty
            best_diversity_bonus = diversity_bonus
            best_redundancy_penalty = redundancy_penalty

    return (
        best_record,
        best_score,
        best_spatial_penalty,
        best_diversity_bonus,
        best_redundancy_penalty,
    )


def _mark_selected_record(
    record: dict[str, object],
    *,
    rank: int,
    score_final: float,
    spatial_penalty: float,
    feature_diversity_bonus_value: float,
    redundancy_penalty: float,
) -> None:
    record["selected"] = True
    record["rank"] = rank
    record["score_final"] = float(score_final)
    record["spatial_penalty"] = float(spatial_penalty)
    record["feature_diversity_bonus"] = float(feature_diversity_bonus_value)
    record["redundancy_penalty"] = float(redundancy_penalty)


def _select_v3_records(
    records: list[dict[str, object]],
    *,
    config: V3ServerQualityConfig,
    slide_dimensions: tuple[int, int],
    min_distance_level0: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    warnings: list[str] = []
    assign_spatial_regions(
        records=records,
        slide_dimensions=slide_dimensions,
        quota_grid=config.quota_grid,
    )

    quality_eligible = [
        record
        for record in records
        if _safe_float(record.get("technical_quality_score")) >= config.min_quality_score
    ]
    if len(quality_eligible) < config.max_patches:
        warnings.append(
            "Fewer candidates passed min_quality_score than max_patches; "
            "selection was completed from the best available scored candidates."
        )
        quality_eligible = list(records)

    threshold = score_quantile(quality_eligible, config.quota_min_score_quantile)
    quota_eligible = [
        record
        for record in quality_eligible
        if _safe_float(record.get("score_raw")) >= threshold
    ]
    selected: list[dict[str, object]] = []
    available = list(quota_eligible)
    regions = sorted({str(record.get("region_id", "")) for record in available if record.get("region_id")})
    region_index = 0

    while available and len(selected) < config.max_patches and regions:
        made_selection = False
        for _ in range(len(regions)):
            region_id = regions[region_index % len(regions)]
            region_index += 1
            regional_candidates = [
                record for record in available
                if record.get("region_id") == region_id
            ]
            if not regional_candidates:
                continue
            (
                best_record,
                best_score,
                best_spatial_penalty,
                best_diversity_bonus,
                best_redundancy_penalty,
            ) = _pick_best_record(
                regional_candidates,
                selected,
                lambda_spatial=config.lambda_spatial,
                min_distance_level0=float(min_distance_level0),
                feature_diversity_weight=config.feature_diversity_weight,
                redundancy_penalty_weight=config.redundancy_penalty_weight,
            )
            if best_record is None:
                continue
            _mark_selected_record(
                best_record,
                rank=len(selected) + 1,
                score_final=best_score,
                spatial_penalty=best_spatial_penalty,
                feature_diversity_bonus_value=best_diversity_bonus,
                redundancy_penalty=best_redundancy_penalty,
            )
            selected.append(best_record)
            available.remove(best_record)
            made_selection = True
            break
        if not made_selection:
            break

    selected_ids = {str(record.get("candidate_id")) for record in selected}
    fill_pool = [
        record
        for record in quality_eligible
        if str(record.get("candidate_id")) not in selected_ids
    ]
    while fill_pool and len(selected) < config.max_patches:
        (
            best_record,
            best_score,
            best_spatial_penalty,
            best_diversity_bonus,
            best_redundancy_penalty,
        ) = _pick_best_record(
            fill_pool,
            selected,
            lambda_spatial=config.lambda_spatial,
            min_distance_level0=float(min_distance_level0),
            feature_diversity_weight=config.feature_diversity_weight,
            redundancy_penalty_weight=config.redundancy_penalty_weight,
        )
        if best_record is None:
            break
        _mark_selected_record(
            best_record,
            rank=len(selected) + 1,
            score_final=best_score,
            spatial_penalty=best_spatial_penalty,
            feature_diversity_bonus_value=best_diversity_bonus,
            redundancy_penalty=best_redundancy_penalty,
        )
        selected.append(best_record)
        fill_pool.remove(best_record)
        selected_ids.add(str(best_record.get("candidate_id")))

    selected_ids = {str(record.get("candidate_id")) for record in selected}
    for record in records:
        if str(record.get("candidate_id")) in selected_ids:
            continue
        score, spatial_penalty, diversity_bonus, redundancy_penalty = _selection_score(
            record,
            selected,
            lambda_spatial=config.lambda_spatial,
            min_distance_level0=float(min_distance_level0),
            feature_diversity_weight=config.feature_diversity_weight,
            redundancy_penalty_weight=config.redundancy_penalty_weight,
        )
        record["selected"] = False
        record["rank"] = ""
        record["score_final"] = float(score)
        record["spatial_penalty"] = float(spatial_penalty)
        record["feature_diversity_bonus"] = float(diversity_bonus)
        record["redundancy_penalty"] = float(redundancy_penalty)

    patches_per_region = Counter(str(record.get("region_id", "")) for record in selected)
    active_region_ids = {
        str(record.get("region_id", ""))
        for record in records
        if record.get("region_id")
    }
    selected_region_ids = set(patches_per_region)
    stats = {
        "score_threshold": threshold,
        "quality_eligible_candidates": len(quality_eligible),
        "quota_eligible_candidates": len(quota_eligible),
        "active_regions": len(active_region_ids),
        "regions_covered": len(selected_region_ids),
        "patches_per_region": dict(sorted(patches_per_region.items())),
        "quota_fill_rate": len(selected_region_ids) / len(active_region_ids)
        if active_region_ids
        else None,
        "quota_grid": config.quota_grid,
        "quota_min_score_quantile": config.quota_min_score_quantile,
        "selected_category_counts": dict(
            sorted(Counter(str(record.get("usefulness_reason", "")) for record in selected).items())
        ),
    }
    return selected, stats, warnings


def _candidate_record_from_patch(
    candidate: PatchCandidate,
    *,
    patch_image: object,
    config: V3ServerQualityConfig,
) -> dict[str, object]:
    features = _compute_v3_raw_features(
        patch_image,
        feature_size=config.feature_size,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "grid_index": candidate.grid_index,
        "x_level0": candidate.x_level0,
        "y_level0": candidate.y_level0,
        "patch_size": candidate.patch_size,
        "width": patch_image.width,
        "height": patch_image.height,
        "thumbnail_tissue_ratio": candidate.thumbnail_tissue_ratio,
        "feature_size": config.feature_size,
        **features,
    }


def _read_cache_rows(cache_path: Path) -> list[dict[str, str]]:
    with cache_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _cached_records_for_candidates(
    cache_rows: list[dict[str, str]],
    *,
    candidates_to_score: list[PatchCandidate],
    config: V3ServerQualityConfig,
    wsi_path: Path,
) -> tuple[list[dict[str, object]] | None, str | None]:
    cache_by_id = {row.get("candidate_id", ""): row for row in cache_rows}
    records: list[dict[str, object]] = []
    required_fields = {
        "candidate_id",
        "x_level0",
        "y_level0",
        "patch_size",
        "thumbnail_tissue_ratio",
        "tissue_ratio",
        "nuclear_signal_rgb",
        "nuclear_signal_hed",
        "visual_entropy",
        "blur_score",
        "artifact_penalty",
        "source_wsi_path",
        "feature_size",
    }
    for candidate in candidates_to_score:
        row = cache_by_id.get(candidate.candidate_id)
        if row is None:
            return None, "Feature cache does not cover the current seeded candidate subset."
        if not required_fields.issubset(row):
            return None, "Feature cache is missing required v3 fields."
        if str(row.get("source_wsi_path")) != str(wsi_path):
            return None, "Feature cache was produced for a different WSI path."
        if int(_safe_float(row.get("feature_size"), -1.0)) != config.feature_size:
            return None, "Feature cache was produced with a different feature_size."
        try:
            record = {
                "candidate_id": candidate.candidate_id,
                "grid_index": candidate.grid_index,
                "x_level0": int(float(row["x_level0"])),
                "y_level0": int(float(row["y_level0"])),
                "patch_size": int(float(row["patch_size"])),
                "width": int(float(row.get("width") or config.patch_size)),
                "height": int(float(row.get("height") or config.patch_size)),
                "thumbnail_tissue_ratio": float(row["thumbnail_tissue_ratio"]),
                "feature_size": int(float(row["feature_size"])),
                "tissue_ratio": float(row["tissue_ratio"]),
                "nuclear_signal": float(row["nuclear_signal_hed"]),
                "nuclear_signal_rgb": float(row["nuclear_signal_rgb"]),
                "nuclear_signal_hed": float(row["nuclear_signal_hed"]),
                "visual_entropy": float(row["visual_entropy"]),
                "blur_score": float(row["blur_score"]),
                "artifact_penalty": float(row["artifact_penalty"]),
            }
        except (TypeError, ValueError) as exc:
            return None, f"Feature cache has invalid numeric values: {exc}"
        records.append(record)
    return records, None


def _update_candidate_row_from_record(row: dict[str, object], record: dict[str, object]) -> None:
    fields_to_copy = [
        "width",
        "height",
        "tissue_ratio",
        "nuclear_signal",
        "nuclear_signal_rgb",
        "nuclear_signal_hed",
        "visual_entropy",
        "blur_score",
        "artifact_penalty",
        "technical_quality_score",
        "heterogeneity_score",
        "cellularity_proxy_score",
        "residual_candidate_proxy",
        "low_cellularity_treated_bed_proxy",
        "tumor_bed_relevance_proxy",
        "usefulness_score",
        "spatial_penalty",
        "feature_diversity_bonus",
        "redundancy_penalty",
        "score_raw",
        "score_final",
        "usefulness_reason",
        *V3_NORMALIZED_FIELDS,
    ]
    row["evaluated"] = True
    row["scored"] = True
    for field_name in fields_to_copy:
        value = record.get(field_name, "")
        row[field_name] = _format_float(value) if isinstance(value, float) else value
    row["region_id"] = record.get("region_id", "")
    row["region_row"] = record.get("region_row", "")
    row["region_col"] = record.get("region_col", "")
    row["quota_grid"] = record.get("quota_grid", row.get("quota_grid", ""))
    row["selected"] = bool(record.get("selected", False))
    row["rank"] = record.get("rank", "")


def _sanitize_numeric_fields(
    records: list[dict[str, object]],
    field_names: list[str],
) -> list[str]:
    """Replace non-finite critical numeric values with 0.0 and report warnings."""
    warnings: list[str] = []
    for field_name in field_names:
        repaired_count = 0
        for record in records:
            value = record.get(field_name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number):
                record[field_name] = 0.0
                repaired_count += 1
        if repaired_count:
            warnings.append(
                f"Replaced {repaired_count} non-finite values in {field_name} with 0.0."
            )
    return warnings


def _score_statistics(records: list[dict[str, object]], field_name: str) -> dict[str, float | None]:
    values = [_safe_float(record.get(field_name), math.nan) for record in records]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _selected_mean(selected_rows: list[dict[str, object]], field_name: str) -> float | None:
    values = [_safe_float(row.get(field_name), math.nan) for row in selected_rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return float(sum(values) / len(values))


def run_v3_server_quality_selection(config: V3ServerQualityConfig) -> dict[str, Any]:
    """Run v3_server_quality and write outputs compatible with selector tooling."""
    start_time = time.perf_counter()
    root_dir = config.root_dir.resolve()
    wsi_path = config.wsi_path.expanduser().resolve()
    output_dir = _resolve_output_dir(config.output_dir, root_dir=root_dir)
    min_distance_level0 = config.min_distance_level0 or config.patch_size
    warnings: list[str] = []

    _validate_config(config=config, wsi_path=wsi_path)

    selected_dir = output_dir / "selected"
    candidate_metadata_path = output_dir / "candidate_metadata.csv"
    selected_metadata_path = output_dir / "selected_metadata.csv"
    summary_path = output_dir / "selection_summary.json"
    method_config_path = output_dir / "method_config.json"
    preview_path = output_dir / "patch_selection_preview.png"
    scored_candidates_path = output_dir / V3_FEATURE_CACHE_FILE

    preexisting_cache_rows: list[dict[str, str]] | None = None
    if config.resume:
        if scored_candidates_path.exists():
            preexisting_cache_rows = _read_cache_rows(scored_candidates_path)
        else:
            warnings.append("--resume was requested, but no scored_candidates.csv cache exists.")

    _prepare_output_dir(
        output_dir=output_dir,
        root_dir=root_dir,
        overwrite=config.overwrite,
    )
    write_json_manifest(
        _method_config(config, min_distance_level0=min_distance_level0),
        method_config_path,
    )

    openslide_module = _import_openslide()
    slide = openslide_module.OpenSlide(str(wsi_path))
    try:
        slide_metadata = _base_slide_metadata(slide)
        slide_dimensions = (
            int(slide_metadata["slide_width"]),
            int(slide_metadata["slide_height"]),
        )

        thumbnail = slide.get_thumbnail(
            (config.thumbnail_max_size, config.thumbnail_max_size)
        ).convert("RGB")
        candidates, num_candidates_generated = generate_tissue_candidates(
            thumbnail=thumbnail,
            slide_dimensions=slide_dimensions,
            patch_size=config.patch_size,
            stride=config.stride,
            min_tissue_ratio=config.min_tissue_ratio,
        )
        candidates_to_score = _select_candidates_to_score(
            candidates,
            seed=config.seed,
            max_candidates_to_score=config.max_candidates_to_score,
        )

        candidate_rows = [
            _candidate_pool_row(
                candidate,
                config=config,
                wsi_path=wsi_path,
                slide_metadata=slide_metadata,
            )
            for candidate in candidates
        ]
        candidate_rows_by_id = {
            str(row["candidate_id"]): row
            for row in candidate_rows
        }

        cache_reused = False
        scored_records: list[dict[str, object]] | None = None
        if preexisting_cache_rows:
            scored_records, cache_error = _cached_records_for_candidates(
                preexisting_cache_rows,
                candidates_to_score=candidates_to_score,
                config=config,
                wsi_path=wsi_path,
            )
            if scored_records is None:
                warnings.append(f"Feature cache was not reused: {cache_error}")
            else:
                cache_reused = True

        if scored_records is None:
            scored_records = []
            for candidate in candidates_to_score:
                patch_image = slide.read_region(
                    (candidate.x_level0, candidate.y_level0),
                    0,
                    (config.patch_size, config.patch_size),
                ).convert("RGB")
                scored_records.append(
                    _candidate_record_from_patch(
                        candidate,
                        patch_image=patch_image,
                        config=config,
                    )
                )
                del patch_image

        _apply_v3_scores(scored_records)
        selected_records, quota_stats, selection_warnings = _select_v3_records(
            records=scored_records,
            config=config,
            slide_dimensions=slide_dimensions,
            min_distance_level0=min_distance_level0,
        )
        warnings.extend(selection_warnings)
        warnings.extend(
            _sanitize_numeric_fields(
                scored_records,
                [*V3_CRITICAL_NUMERIC_FIELDS, *V3_NORMALIZED_FIELDS],
            )
        )

        for record in scored_records:
            row = candidate_rows_by_id[str(record["candidate_id"])]
            _update_candidate_row_from_record(row, record)

        selected_rows: list[dict[str, object]] = []
        for record in sorted(selected_records, key=lambda item: int(item["rank"])):
            row = candidate_rows_by_id[str(record["candidate_id"])]
            patch_id = f"patch_{len(selected_rows):04d}_x{row['x_level0']}_y{row['y_level0']}"
            filename = f"{patch_id}.png"
            patch_path = selected_dir / filename
            patch_image = slide.read_region(
                (int(row["x_level0"]), int(row["y_level0"])),
                0,
                (config.patch_size, config.patch_size),
            ).convert("RGB")
            patch_image.save(patch_path)
            row["filename"] = filename
            row["patch_id"] = patch_id
            row["patch_path"] = str(patch_path)
            selected_rows.append(_selected_row(row))
            del patch_image

        write_csv_manifest(
            rows=candidate_rows,
            output_path=candidate_metadata_path,
            fieldnames=CANDIDATE_METADATA_FIELDS,
        )
        write_csv_manifest(
            rows=selected_rows,
            output_path=selected_metadata_path,
            fieldnames=SELECTED_METADATA_FIELDS,
        )

        scored_candidate_rows = [
            row for row in candidate_rows
            if row.get("scored") in (True, "True", "true", "1")
        ]
        if config.cache_features or config.output_mode in {"debug", "full"}:
            for row in scored_candidate_rows:
                row["feature_size"] = config.feature_size
            write_csv_manifest(
                rows=scored_candidate_rows,
                output_path=scored_candidates_path,
                fieldnames=SCORED_CANDIDATE_FIELDS,
            )

        save_wsi_patch_selection_preview(
            thumbnail=thumbnail,
            candidate_rows=scored_candidate_rows,
            slide_dimensions=slide_dimensions,
            output_path=preview_path,
        )

        score_fields = [
            "technical_quality_score",
            "residual_candidate_proxy",
            "low_cellularity_treated_bed_proxy",
            "tumor_bed_relevance_proxy",
            "usefulness_score",
            "score_raw",
            "score_final",
        ]
        score_statistics = {
            field_name: _score_statistics(scored_records, field_name)
            for field_name in score_fields
        }
        summary: dict[str, Any] = {
            "status": "completed_with_warnings" if warnings else "completed",
            "selector": config.selector,
            "selector_name": config.selector,
            "version": V3_SELECTOR_VERSION,
            "wsi_path": str(wsi_path),
            "output_dir": str(output_dir),
            "patch_size": config.patch_size,
            "stride": config.stride,
            "max_patches": config.max_patches,
            "min_tissue_ratio": config.min_tissue_ratio,
            "seed": config.seed,
            "thumbnail_max_size": config.thumbnail_max_size,
            "feature_size": config.feature_size,
            "max_candidates_to_score": config.max_candidates_to_score,
            "lambda_spatial": config.lambda_spatial,
            "min_distance_level0": min_distance_level0,
            "quota_grid": config.quota_grid,
            "quota_min_score_quantile": config.quota_min_score_quantile,
            "feature_diversity_weight": config.feature_diversity_weight,
            "redundancy_penalty_weight": config.redundancy_penalty_weight,
            "min_quality_score": config.min_quality_score,
            "resume": config.resume,
            "cache_features": config.cache_features,
            "feature_cache_reused": cache_reused,
            "output_mode": config.output_mode,
            **slide_metadata,
            "num_candidates_generated": num_candidates_generated,
            "num_thumbnail_candidates_passing_mask": len(candidates),
            "num_candidate_rows_written": len(candidate_rows),
            "num_candidates_scored": len(scored_records),
            "num_candidates_evaluated": len(scored_records),
            "num_selected": len(selected_rows),
            "score_statistics": score_statistics,
            "selected_category_counts": quota_stats.get("selected_category_counts", {}),
            "spatial_coverage": {
                "regions_covered": quota_stats.get("regions_covered"),
                "active_regions": quota_stats.get("active_regions"),
                "patches_per_region": quota_stats.get("patches_per_region"),
                "quota_fill_rate": quota_stats.get("quota_fill_rate"),
            },
            "quality_eligible_candidates": quota_stats.get("quality_eligible_candidates"),
            "quota_eligible_candidates": quota_stats.get("quota_eligible_candidates"),
            "quota_score_threshold": quota_stats.get("score_threshold"),
            "mean_technical_quality_score_selected": _selected_mean(
                selected_rows,
                "technical_quality_score",
            ),
            "mean_residual_candidate_proxy_selected": _selected_mean(
                selected_rows,
                "residual_candidate_proxy",
            ),
            "mean_low_cellularity_treated_bed_proxy_selected": _selected_mean(
                selected_rows,
                "low_cellularity_treated_bed_proxy",
            ),
            "mean_tumor_bed_relevance_proxy_selected": _selected_mean(
                selected_rows,
                "tumor_bed_relevance_proxy",
            ),
            "mean_usefulness_score_selected": _selected_mean(
                selected_rows,
                "usefulness_score",
            ),
            "mean_score_final_selected": _selected_mean(selected_rows, "score_final"),
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "candidate_metadata_csv": str(candidate_metadata_path),
            "selected_metadata_csv": str(selected_metadata_path),
            "scored_candidates_csv": str(scored_candidates_path)
            if scored_candidates_path.exists()
            else None,
            "method_config_json": str(method_config_path),
            "preview_image": str(preview_path),
            "selected_dir": str(selected_dir),
            "candidate_pool": CANDIDATE_POOL,
            "candidate_metadata_semantics": CANDIDATE_METADATA_SEMANTICS,
            "preview_shows": V3_PREVIEW_SHOWS,
            "candidate_ordering": V3_CANDIDATE_ORDERING,
            "tissue_mask_method": TISSUE_MASK_METHOD,
            "nuclear_signal_rgb_method": RGB_NUCLEAR_SIGNAL_METHOD,
            "nuclear_signal_hed_method": HED_NUCLEAR_SIGNAL_METHOD,
            "visual_entropy_method": VISUAL_ENTROPY_METHOD,
            "blur_score_method": BLUR_SCORE_METHOD,
            "artifact_penalty_method": ARTIFACT_PENALTY_METHOD,
            "weights": V3_WEIGHTS,
            "useful_patch_definition": USEFUL_PATCH_DEFINITION,
            "no_deep_learning_used_for_selection": True,
            "segmentation_model_used_for_selection": False,
            "selection_model_note": NO_MODEL_SELECTION_NOTE,
            "clinical_warning": CLINICAL_WARNING,
            "warnings": warnings,
        }
        write_json_manifest(summary, summary_path)
        return summary
    finally:
        slide.close()
