"""Medical-image-feature gated embedding-assisted patch selector.

v4_1_medical_embedding_assisted keeps v3_server_quality as the dominant
technical signal, adds classical medical-image processing proxies, and uses UNI
only as a morphology embedding reranker inside technically strong candidates.
It does not run segmentation, diagnose, calculate RCB, or validate clinically.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.wsi_patch_extraction import (
    SUPPORTED_WSI_EXTENSIONS,
    _import_openslide,
)
from src.selection.diversity import (
    assign_spatial_regions,
    feature_diversity_bonus,
    proximity_penalty,
)
from src.selection.embedding_scoring import (
    UNI_BACKEND_MISSING_MESSAGE,
    cosine_distance_to_selected,
    load_embedding_cache,
    validate_embedding_cache,
    write_embedding_cache,
)
from src.selection.manifests import (
    CANDIDATE_METADATA_FIELDS,
    SELECTED_METADATA_FIELDS,
    utc_now_iso,
    write_csv_manifest,
    write_json_manifest,
)
from src.selection.medical_image_features import (
    MEDICAL_IMAGE_FEATURE_FIELDS,
    MEDICAL_IMAGE_FEATURE_METHODS,
    MEDICAL_SCORE_FIELDS,
    compute_medical_image_features,
)
from src.selection.previews import save_wsi_patch_selection_preview
from src.selection.quality_filters import (
    ARTIFACT_PENALTY_METHOD,
    BLUR_SCORE_METHOD,
    HED_NUCLEAR_SIGNAL_METHOD,
    RGB_NUCLEAR_SIGNAL_METHOD,
    VISUAL_ENTROPY_METHOD,
)
from src.selection.tiatoolbox_baseline import (
    CLINICAL_WARNING,
    TIATOOLBOX_CANDIDATE_METADATA_SEMANTICS,
    TIATOOLBOX_CANDIDATE_POOL,
    TIATOOLBOX_EXTRACTION_BACKEND,
    TIATOOLBOX_EXTRACTOR_NAME,
    TIATOOLBOX_INPUT_MASK,
    TIATOOLBOX_TISSUE_MASK_METHOD,
    _base_slide_metadata,
    _build_tiatoolbox_extractor,
    _candidate_pool_row as _tiatoolbox_candidate_pool_row,
    _candidates_from_extractor,
    _prepare_output_dir,
    _resolve_output_dir,
)
from src.selection.v3_server_quality import (
    NO_MODEL_SELECTION_NOTE,
    USEFUL_PATCH_DEFINITION,
    V3_CRITICAL_NUMERIC_FIELDS,
    V3_FEATURE_DIVERSITY_FIELDS,
    V3_NORMALIZED_FIELDS,
    V3_WEIGHTS,
    _apply_v3_scores,
    _candidate_record_from_patch,
    _format_float,
    _safe_float,
    _sanitize_numeric_fields,
    _score_statistics,
    _select_candidates_to_score,
    _selected_mean,
    _update_candidate_row_from_record,
)
from src.selection.v4_embedding_assisted import (
    EMBEDDING_CACHE_FILE,
    EMBEDDING_CACHE_METADATA_FILE,
    EMBEDDING_CLUSTER_SUMMARY_FILE,
    EMBEDDING_SELECTION_NOTE,
    SCORED_CANDIDATES_FILE,
    V4_EMBEDDING_FIELDS,
    _apply_embedding_metrics,
    _compute_embeddings_for_candidates,
    _embedding_cache_paths,
    _mean_record_value,
    _write_cluster_summary,
)


V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME = "v4_1_medical_embedding_assisted"
V41_SELECTOR_VERSION = "v4_1_medical_embedding_assisted_1.0"
V41_CANDIDATE_ORDERING = (
    "tiatoolbox_otsu_seeded_shuffle_then_v3_medical_quality_filter_then_uni_embedding_rerank"
)
V41_PREVIEW_SHOWS = "scored_candidates_with_v3_medical_embedding_selected_highlighted"
V41_MEDICAL_RERANK_MODES = {"top_v3_then_embedding"}

V41_WEIGHTS = {
    "v3_base": 1.00,
    "medical_image_quality": 0.25,
    "medical_image_utility": 0.25,
    "embedding_diversity": 0.08,
    "embedding_redundancy": 0.08,
    "cluster_balance": 0.05,
    "representative_cluster": 0.05,
}

V41_CRITICAL_NUMERIC_FIELDS = [
    *V3_CRITICAL_NUMERIC_FIELDS,
    *MEDICAL_SCORE_FIELDS,
    "score_v3_base",
    "embedding_distance_to_cluster_centroid",
    "embedding_representativeness_score",
    "embedding_novelty_score",
    "embedding_diversity_bonus",
    "embedding_redundancy_penalty",
    "morphology_diversity_score",
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
class V41MedicalEmbeddingAssistedConfig:
    """Configuration for v4_1_medical_embedding_assisted."""

    wsi_path: Path
    output_dir: Path
    root_dir: Path
    selector: str = V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME
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
    feature_diversity_weight: float = 0.10
    redundancy_penalty_weight: float = 0.10
    min_quality_score: float = 0.15
    resume: bool = False
    cache_features: bool = False
    output_mode: str = "debug"
    embedding_backend: str = "uni"
    embedding_model_name: str = "UNI"
    embedding_model_path: Path | None = None
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_num_workers: int = 2
    embedding_cache_path: Path | None = None
    cache_embeddings: bool = True
    reuse_embedding_cache: bool = True
    embedding_dim: int | None = None
    embedding_distance_metric: str = "cosine"
    embedding_diversity_weight: float = 0.08
    embedding_redundancy_weight: float = 0.08
    embedding_cluster_count: int = 8
    cluster_balance_weight: float = 0.05
    representative_cluster_weight: float = 0.05
    medical_min_quality_score: float = 0.50
    medical_min_utility_score: float = 0.45
    min_score_v3_base_quantile: float = 0.80
    medical_top_k_candidates: int | None = None
    medical_top_quantile: float = 0.20
    medical_artifact_max: float = 0.12
    medical_rerank_mode: str = "top_v3_then_embedding"


def _validate_config(config: V41MedicalEmbeddingAssistedConfig, wsi_path: Path) -> None:
    if config.selector != V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME:
        raise NotImplementedError(
            f"Selector '{config.selector}' is not handled by v4_1_medical_embedding_assisted."
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
    if config.embedding_backend != "uni":
        raise ValueError("--embedding-backend currently supports only uni.")
    if config.embedding_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("--embedding-device must be auto, cpu, cuda, or mps.")
    if config.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive.")
    if config.embedding_num_workers < 0:
        raise ValueError("--embedding-num-workers must be >= 0.")
    if config.embedding_dim is not None and config.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive when provided.")
    if config.embedding_distance_metric != "cosine":
        raise ValueError("--embedding-distance-metric currently supports only cosine.")
    if config.embedding_cluster_count <= 0:
        raise ValueError("--embedding-cluster-count must be positive.")
    for field_name, value in {
        "embedding_diversity_weight": config.embedding_diversity_weight,
        "embedding_redundancy_weight": config.embedding_redundancy_weight,
        "cluster_balance_weight": config.cluster_balance_weight,
        "representative_cluster_weight": config.representative_cluster_weight,
        "medical_min_quality_score": config.medical_min_quality_score,
        "medical_min_utility_score": config.medical_min_utility_score,
        "medical_artifact_max": config.medical_artifact_max,
    }.items():
        if value < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be >= 0.")
    if not 0 <= config.min_score_v3_base_quantile <= 1:
        raise ValueError("--min-score-v3-base-quantile must be between 0 and 1.")
    if not 0 < config.medical_top_quantile <= 1:
        raise ValueError("--medical-top-quantile must be in (0, 1].")
    if config.medical_top_k_candidates is not None and config.medical_top_k_candidates <= 0:
        raise ValueError("--medical-top-k-candidates must be positive when provided.")
    if config.medical_rerank_mode not in V41_MEDICAL_RERANK_MODES:
        allowed = ", ".join(sorted(V41_MEDICAL_RERANK_MODES))
        raise ValueError(f"--medical-rerank-mode must be one of: {allowed}.")
    if wsi_path.suffix.lower() not in SUPPORTED_WSI_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_WSI_EXTENSIONS))
        raise ValueError(f"Unsupported WSI extension '{wsi_path.suffix}'. Use one of: {allowed}.")
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")


def _method_config(
    config: V41MedicalEmbeddingAssistedConfig,
    *,
    min_distance_level0: int,
    embedding_cache_path: Path,
    tiatoolbox_version: str | None = None,
) -> dict[str, object]:
    return {
        "selector": config.selector,
        "selector_name": config.selector,
        "version": V41_SELECTOR_VERSION,
        "candidate_pool": TIATOOLBOX_CANDIDATE_POOL,
        "candidate_metadata_semantics": TIATOOLBOX_CANDIDATE_METADATA_SEMANTICS,
        "candidate_pool_definition": (
            "TIAToolbox sliding-window candidates passing Otsu input mask and min_mask_ratio."
        ),
        "candidate_pool_shared_with_baseline": True,
        "shared_candidate_pool_selector": "baseline_tiatoolbox",
        "candidate_ordering": V41_CANDIDATE_ORDERING,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "min_mask_ratio": config.min_tissue_ratio,
        "seed": config.seed,
        "thumbnail_max_size": config.thumbnail_max_size,
        "feature_size": config.feature_size,
        "max_candidates_to_score": config.max_candidates_to_score,
        "v3_weights": V3_WEIGHTS,
        "v4_1_weights": {
            **V41_WEIGHTS,
            "embedding_diversity": config.embedding_diversity_weight,
            "embedding_redundancy": config.embedding_redundancy_weight,
            "cluster_balance": config.cluster_balance_weight,
            "representative_cluster": config.representative_cluster_weight,
        },
        "lambda_spatial": config.lambda_spatial,
        "min_distance_level0": min_distance_level0,
        "quota_grid": config.quota_grid,
        "quota_min_score_quantile": config.quota_min_score_quantile,
        "feature_diversity_weight": config.feature_diversity_weight,
        "redundancy_penalty_weight": config.redundancy_penalty_weight,
        "min_quality_score": config.min_quality_score,
        "medical_min_quality_score": config.medical_min_quality_score,
        "medical_min_utility_score": config.medical_min_utility_score,
        "min_score_v3_base_quantile": config.min_score_v3_base_quantile,
        "medical_top_k_candidates": config.medical_top_k_candidates,
        "medical_top_quantile": config.medical_top_quantile,
        "medical_artifact_max": config.medical_artifact_max,
        "medical_rerank_mode": config.medical_rerank_mode,
        "output_mode": config.output_mode,
        "embedding_backend": config.embedding_backend,
        "embedding_model_name": config.embedding_model_name,
        "embedding_model_path": str(config.embedding_model_path) if config.embedding_model_path else None,
        "embedding_device": config.embedding_device,
        "embedding_batch_size": config.embedding_batch_size,
        "embedding_num_workers": config.embedding_num_workers,
        "embedding_cache_path": str(embedding_cache_path),
        "cache_embeddings": config.cache_embeddings,
        "reuse_embedding_cache": config.reuse_embedding_cache,
        "embedding_dim": config.embedding_dim,
        "embedding_distance_metric": config.embedding_distance_metric,
        "embedding_cluster_count": config.embedding_cluster_count,
        "medical_image_feature_methods": MEDICAL_IMAGE_FEATURE_METHODS,
        "useful_patch_definition": USEFUL_PATCH_DEFINITION,
        "embedding_selection_note": EMBEDDING_SELECTION_NOTE,
        "medical_selection_note": (
            "Classical medical-image features are technical proxies for image quality, "
            "texture, stain dynamics, and pseudo-cellularity. They are not clinical labels."
        ),
        "no_deep_learning_used_for_selection": False,
        "segmentation_model_used_for_selection": False,
        "selection_model_note": NO_MODEL_SELECTION_NOTE,
        "extraction_backend": TIATOOLBOX_EXTRACTION_BACKEND,
        "tiatoolbox_version": tiatoolbox_version,
        "tiatoolbox_extractor": TIATOOLBOX_EXTRACTOR_NAME,
        "input_mask": TIATOOLBOX_INPUT_MASK,
        "tissue_mask_method": TIATOOLBOX_TISSUE_MASK_METHOD,
        "nuclear_signal_rgb_method": RGB_NUCLEAR_SIGNAL_METHOD,
        "nuclear_signal_hed_method": HED_NUCLEAR_SIGNAL_METHOD,
        "visual_entropy_method": VISUAL_ENTROPY_METHOD,
        "blur_score_method": BLUR_SCORE_METHOD,
        "artifact_penalty_method": ARTIFACT_PENALTY_METHOD,
        "clinical_warning": CLINICAL_WARNING,
        "created_at": utc_now_iso(),
    }


def _v41_candidate_pool_row(
    candidate: object,
    *,
    config: V41MedicalEmbeddingAssistedConfig,
    wsi_path: Path,
    slide_metadata: dict[str, Any],
) -> dict[str, object]:
    row = _tiatoolbox_candidate_pool_row(
        candidate,
        config=config,
        wsi_path=wsi_path,
        slide_metadata=slide_metadata,
    )
    row.update(
        {
            "scored": False,
            "nuclear_proxy": "hed_deconvolution",
            "score_raw": "",
            "score_final": "",
            "usefulness_reason": "",
            "quota_grid": config.quota_grid,
            "spatial_strategy": "quotas",
            "diversity_strategy": "farthest_feature",
            "embedding_backend": config.embedding_backend,
            "embedding_model_name": config.embedding_model_name,
            "selector": config.selector,
            "selection_method": config.selector,
            "tissue_mask_method": TIATOOLBOX_TISSUE_MASK_METHOD,
            "clinical_warning": CLINICAL_WARNING,
        }
    )
    return row


def _score_quantile_by_field(
    records: list[dict[str, object]],
    field_name: str,
    quantile: float,
) -> float:
    if not records:
        return 0.0
    values = sorted(_safe_float(record.get(field_name), 0.0) for record in records)
    clipped = max(0.0, min(1.0, quantile))
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * clipped))))
    return float(values[index])


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _output_local_embedding_cache_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / EMBEDDING_CACHE_FILE, output_dir / EMBEDDING_CACHE_METADATA_FILE


def _apply_medical_features(
    records: list[dict[str, object]],
    *,
    slide: object,
    candidates_to_score: list[object],
    config: V41MedicalEmbeddingAssistedConfig,
) -> None:
    for record, candidate in zip(records, candidates_to_score):
        patch_image = slide.read_region(
            (candidate.x_level0, candidate.y_level0),
            0,
            (config.patch_size, config.patch_size),
        ).convert("RGB")
        features = compute_medical_image_features(
            patch_image,
            feature_size=config.feature_size,
            tumor_bed_relevance_proxy=_safe_float(record.get("tumor_bed_relevance_proxy")),
        )
        record.update(features)
        del patch_image


def _apply_v41_static_scores(records: list[dict[str, object]]) -> None:
    for record in records:
        score_v3_base = _safe_float(record.get("score_v3_base", record.get("score_raw")))
        medical_quality = _safe_float(record.get("medical_image_quality_score"))
        medical_utility = _safe_float(record.get("medical_image_utility_score"))
        record["score_v3_base"] = score_v3_base
        record["score_raw"] = float(
            V41_WEIGHTS["v3_base"] * score_v3_base
            + V41_WEIGHTS["medical_image_quality"] * medical_quality
            + V41_WEIGHTS["medical_image_utility"] * medical_utility
        )


def _load_or_compute_embeddings_v41(
    *,
    slide: object,
    candidates_to_score: list[object],
    config: V41MedicalEmbeddingAssistedConfig,
    output_dir: Path,
    warnings: list[str],
) -> tuple[np.ndarray, bool, Path, Path]:
    candidate_ids = [candidate.candidate_id for candidate in candidates_to_score]
    cache_path, metadata_path = _embedding_cache_paths(config, output_dir)
    if config.reuse_embedding_cache and cache_path.exists() and metadata_path.exists():
        embeddings, cached_candidate_ids, metadata = load_embedding_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
        )
        cache_error = validate_embedding_cache(
            candidate_ids=candidate_ids,
            embeddings=embeddings,
            cached_candidate_ids=cached_candidate_ids,
            metadata=metadata,
            embedding_backend=config.embedding_backend,
            embedding_model_name=config.embedding_model_name,
            embedding_distance_metric=config.embedding_distance_metric,
            expected_dim=config.embedding_dim,
        )
        if cache_error is None:
            return embeddings, True, cache_path, metadata_path
        warnings.append(f"Embedding cache was not reused: {cache_error}")

    if config.embedding_model_path is None:
        raise RuntimeError(UNI_BACKEND_MISSING_MESSAGE)

    embeddings = _compute_embeddings_for_candidates(
        slide=slide,
        candidates=candidates_to_score,
        config=config,
    )
    write_cache_path = cache_path
    write_metadata_path = metadata_path
    if config.embedding_cache_path is not None and not _is_relative_to(cache_path, output_dir):
        write_cache_path, write_metadata_path = _output_local_embedding_cache_paths(output_dir)
        warnings.append(
            "External embedding cache path was not overwritten after cache mismatch; "
            "new embeddings were written to the v4.1 output directory."
        )
    if config.cache_embeddings:
        write_embedding_cache(
            embeddings=embeddings,
            candidate_ids=candidate_ids,
            cache_path=write_cache_path,
            metadata_path=write_metadata_path,
            metadata={
                "selector": config.selector,
                "embedding_backend": config.embedding_backend,
                "embedding_model_name": config.embedding_model_name,
                "embedding_model_path": str(config.embedding_model_path),
                "embedding_distance_metric": config.embedding_distance_metric,
                "feature_size": config.feature_size,
                "patch_size": config.patch_size,
                "created_at": utc_now_iso(),
            },
        )
    return embeddings, False, write_cache_path, write_metadata_path


def _selected_row(candidate_row: dict[str, object]) -> dict[str, object]:
    return {
        field_name: candidate_row.get(field_name, "")
        for field_name in SELECTED_METADATA_FIELDS
    }


def _update_candidate_row_with_v41(
    row: dict[str, object],
    record: dict[str, object],
    *,
    embedding_backend: str,
    embedding_model_name: str,
    embedding_dim: int | None,
    embedding_cache_used: bool,
) -> None:
    _update_candidate_row_from_record(row, record)
    row["selector"] = V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME
    row["selection_method"] = V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME
    row["score_v3_base"] = _format_float(record.get("score_v3_base", record.get("score_raw", 0.0)))
    row["embedding_backend"] = embedding_backend
    row["embedding_model_name"] = embedding_model_name
    row["embedding_dim"] = embedding_dim if embedding_dim is not None else ""
    row["embedding_cache_used"] = embedding_cache_used
    for field_name in [*MEDICAL_IMAGE_FEATURE_FIELDS, *V4_EMBEDDING_FIELDS]:
        if field_name in {"embedding_backend", "embedding_model_name", "embedding_dim", "embedding_cache_used"}:
            continue
        value = record.get(field_name, "")
        row[field_name] = _format_float(value) if isinstance(value, float) else value


def _cluster_balance_score(
    record: dict[str, object],
    selected_records: list[dict[str, object]],
) -> float:
    cluster_id = str(record.get("embedding_cluster_id", ""))
    selected_in_cluster = sum(
        1 for selected in selected_records
        if str(selected.get("embedding_cluster_id", "")) == cluster_id
    )
    return float(1.0 / (1.0 + selected_in_cluster))


def _v41_score(
    record: dict[str, object],
    selected_records: list[dict[str, object]],
    selected_embeddings: np.ndarray,
    *,
    embedding: np.ndarray,
    config: V41MedicalEmbeddingAssistedConfig,
    min_distance_level0: float,
) -> tuple[float, dict[str, float]]:
    spatial_penalty = proximity_penalty(
        candidate=record,
        selected_records=selected_records,
        min_distance_level0=min_distance_level0,
    )
    feature_bonus = (
        feature_diversity_bonus(
            candidate=record,
            selected_records=selected_records,
            feature_names=[
                *V3_FEATURE_DIVERSITY_FIELDS,
                "medical_image_quality_score",
                "medical_image_utility_score",
                "medical_texture_score",
                "medical_pseudo_cellularity_score",
            ],
        )
        if selected_records
        else 0.0
    )
    feature_redundancy = (1.0 - min(1.0, feature_bonus)) if selected_records else 0.0
    embedding_distance, embedding_similarity = cosine_distance_to_selected(
        candidate_embedding=embedding,
        selected_embeddings=selected_embeddings,
    )
    embedding_diversity_bonus = embedding_distance if selected_records else 0.0
    embedding_redundancy_penalty = max(0.0, embedding_similarity) if selected_records else 0.0
    novelty_score = embedding_diversity_bonus
    cluster_balance = _cluster_balance_score(record, selected_records)
    representative = _safe_float(record.get("embedding_representativeness_score"))
    morphology_diversity_score = max(
        0.0,
        min(1.0, 0.45 * novelty_score + 0.30 * cluster_balance + 0.25 * representative),
    )
    score_final = (
        _safe_float(record.get("score_raw"))
        - config.lambda_spatial * spatial_penalty
        - config.redundancy_penalty_weight * feature_redundancy
        + config.feature_diversity_weight * feature_bonus
        + config.embedding_diversity_weight * embedding_diversity_bonus
        + config.cluster_balance_weight * cluster_balance
        + config.representative_cluster_weight * representative
        - config.embedding_redundancy_weight * embedding_redundancy_penalty
    )
    return float(score_final), {
        "spatial_penalty": float(spatial_penalty),
        "feature_diversity_bonus": float(feature_bonus),
        "redundancy_penalty": float(feature_redundancy),
        "embedding_novelty_score": float(novelty_score),
        "embedding_diversity_bonus": float(embedding_diversity_bonus),
        "embedding_redundancy_penalty": float(embedding_redundancy_penalty),
        "morphology_diversity_score": float(morphology_diversity_score),
    }


def _pick_best_record(
    available: list[dict[str, object]],
    selected: list[dict[str, object]],
    selected_embeddings: np.ndarray,
    embeddings_by_id: dict[str, np.ndarray],
    *,
    config: V41MedicalEmbeddingAssistedConfig,
    min_distance_level0: float,
) -> tuple[dict[str, object] | None, float, dict[str, float]]:
    best_record: dict[str, object] | None = None
    best_score = -math.inf
    best_components: dict[str, float] = {}
    for record in available:
        candidate_id = str(record["candidate_id"])
        score, components = _v41_score(
            record,
            selected,
            selected_embeddings,
            embedding=embeddings_by_id[candidate_id],
            config=config,
            min_distance_level0=min_distance_level0,
        )
        tie_breaker = candidate_id
        best_tie_breaker = str(best_record.get("candidate_id", "")) if best_record else ""
        if score > best_score or (
            math.isclose(score, best_score) and tie_breaker < best_tie_breaker
        ):
            best_record = record
            best_score = score
            best_components = components
    return best_record, best_score, best_components


def _mark_record(
    record: dict[str, object],
    *,
    rank: int,
    score_final: float,
    components: dict[str, float],
) -> None:
    record["selected"] = True
    record["rank"] = rank
    record["score_final"] = float(score_final)
    for key, value in components.items():
        record[key] = float(value)


def _top_medical_records(
    records: list[dict[str, object]],
    *,
    top_quantile: float,
    top_k: int | None,
) -> list[dict[str, object]]:
    if not records:
        return []
    threshold = _score_quantile_by_field(
        records,
        "medical_image_utility_score",
        max(0.0, 1.0 - top_quantile),
    )
    top_records = [
        record
        for record in records
        if _safe_float(record.get("medical_image_utility_score")) >= threshold
    ]
    if top_k is not None and len(top_records) > top_k:
        top_records = sorted(
            top_records,
            key=lambda record: (
                -_safe_float(record.get("medical_image_utility_score")),
                -_safe_float(record.get("score_v3_base")),
                str(record.get("candidate_id", "")),
            ),
        )[:top_k]
    return top_records


def _eligible_records(
    records: list[dict[str, object]],
    *,
    config: V41MedicalEmbeddingAssistedConfig,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    warnings: list[str] = []
    v3_threshold = _score_quantile_by_field(
        records,
        "score_v3_base",
        config.min_score_v3_base_quantile,
    )
    v3_top = [
        record for record in records
        if _safe_float(record.get("score_v3_base")) >= v3_threshold
    ]
    medical_top = _top_medical_records(
        v3_top,
        top_quantile=config.medical_top_quantile,
        top_k=config.medical_top_k_candidates,
    )
    strict = [
        record for record in medical_top
        if _safe_float(record.get("medical_image_quality_score")) >= config.medical_min_quality_score
        and _safe_float(record.get("medical_image_utility_score")) >= config.medical_min_utility_score
        and _safe_float(record.get("medical_artifact_penalty")) <= config.medical_artifact_max
    ]
    chosen_stage = "strict"
    eligible = strict
    relaxed = [
        record for record in v3_top
        if _safe_float(record.get("medical_image_quality_score")) >= max(0.0, config.medical_min_quality_score - 0.10)
        and _safe_float(record.get("medical_image_utility_score")) >= max(0.0, config.medical_min_utility_score - 0.10)
        and _safe_float(record.get("medical_artifact_penalty")) <= min(1.0, config.medical_artifact_max + 0.05)
    ]
    if len(eligible) < config.max_patches and len(relaxed) > len(eligible):
        warnings.append(
            "Strict v4.1 medical eligibility had fewer candidates than max_patches; "
            "relaxed quality/utility/artifact thresholds were used."
        )
        chosen_stage = "relaxed_thresholds"
        eligible = relaxed
    if len(eligible) < config.max_patches and len(medical_top) > len(eligible):
        warnings.append(
            "Relaxed v4.1 eligibility still had fewer candidates than max_patches; "
            "top medical-utility candidates from the v3 pool were used."
        )
        chosen_stage = "top_v3_medical_pool"
        eligible = medical_top
    if len(eligible) < config.max_patches:
        warnings.append(
            "Top v3/medical pool still had fewer candidates than max_patches; "
            "selection can fill from all scored candidates."
        )
        chosen_stage = "all_scored_fallback"
        eligible = list(records)
    stats = {
        "v3_base_threshold": v3_threshold,
        "v3_top_candidates": len(v3_top),
        "medical_top_candidates": len(medical_top),
        "strict_eligible_candidates": len(strict),
        "relaxed_eligible_candidates": len(relaxed),
        "eligible_candidates": len(eligible),
        "eligibility_stage": chosen_stage,
    }
    return eligible, stats, warnings


def _select_v41_records(
    records: list[dict[str, object]],
    embeddings: np.ndarray,
    *,
    config: V41MedicalEmbeddingAssistedConfig,
    slide_dimensions: tuple[int, int],
    min_distance_level0: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    warnings: list[str] = []
    assign_spatial_regions(
        records=records,
        slide_dimensions=slide_dimensions,
        quota_grid=config.quota_grid,
    )
    candidate_ids = [str(record["candidate_id"]) for record in records]
    embeddings_by_id = {
        candidate_id: embeddings[index]
        for index, candidate_id in enumerate(candidate_ids)
    }
    eligible, eligibility_stats, eligibility_warnings = _eligible_records(records, config=config)
    warnings.extend(eligibility_warnings)

    selected: list[dict[str, object]] = []
    selected_embedding_rows: list[np.ndarray] = []
    selected_embeddings = np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    available = list(eligible)
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
            best_record, best_score, components = _pick_best_record(
                regional_candidates,
                selected,
                selected_embeddings,
                embeddings_by_id,
                config=config,
                min_distance_level0=float(min_distance_level0),
            )
            if best_record is None:
                continue
            _mark_record(
                best_record,
                rank=len(selected) + 1,
                score_final=best_score,
                components=components,
            )
            selected.append(best_record)
            selected_embedding_rows.append(embeddings_by_id[str(best_record["candidate_id"])])
            selected_embeddings = np.asarray(selected_embedding_rows, dtype=np.float32)
            available.remove(best_record)
            made_selection = True
            break
        if not made_selection:
            break

    selected_ids = {str(record["candidate_id"]) for record in selected}
    fill_pool = [
        record
        for record in eligible
        if str(record["candidate_id"]) not in selected_ids
    ]
    if len(selected) < config.max_patches and not fill_pool:
        fill_pool = [
            record
            for record in records
            if str(record["candidate_id"]) not in selected_ids
        ]
    while fill_pool and len(selected) < config.max_patches:
        best_record, best_score, components = _pick_best_record(
            fill_pool,
            selected,
            selected_embeddings,
            embeddings_by_id,
            config=config,
            min_distance_level0=float(min_distance_level0),
        )
        if best_record is None:
            break
        _mark_record(
            best_record,
            rank=len(selected) + 1,
            score_final=best_score,
            components=components,
        )
        selected.append(best_record)
        selected_embedding_rows.append(embeddings_by_id[str(best_record["candidate_id"])])
        selected_embeddings = np.asarray(selected_embedding_rows, dtype=np.float32)
        fill_pool.remove(best_record)
        selected_ids.add(str(best_record["candidate_id"]))

    selected_ids = {str(record["candidate_id"]) for record in selected}
    for record in records:
        if str(record["candidate_id"]) in selected_ids:
            continue
        score, components = _v41_score(
            record,
            selected,
            selected_embeddings,
            embedding=embeddings_by_id[str(record["candidate_id"])],
            config=config,
            min_distance_level0=float(min_distance_level0),
        )
        record["selected"] = False
        record["rank"] = ""
        record["score_final"] = float(score)
        for key, value in components.items():
            record[key] = float(value)

    patches_per_region = Counter(str(record.get("region_id", "")) for record in selected)
    selected_cluster_counts = Counter(str(record.get("embedding_cluster_id", "")) for record in selected)
    candidate_cluster_counts = Counter(str(record.get("embedding_cluster_id", "")) for record in records)
    active_region_ids = {
        str(record.get("region_id", ""))
        for record in records
        if record.get("region_id")
    }
    selected_region_ids = set(patches_per_region)
    stats = {
        **eligibility_stats,
        "active_regions": len(active_region_ids),
        "regions_covered": len(selected_region_ids),
        "patches_per_region": dict(sorted(patches_per_region.items())),
        "quota_fill_rate": len(selected_region_ids) / len(active_region_ids)
        if active_region_ids
        else None,
        "selected_clusters": dict(sorted(selected_cluster_counts.items())),
        "candidate_clusters": dict(sorted(candidate_cluster_counts.items())),
        "selected_category_counts": dict(
            sorted(Counter(str(record.get("usefulness_reason", "")) for record in selected).items())
        ),
    }
    return selected, stats, warnings


def run_v4_1_medical_embedding_assisted_selection(
    config: V41MedicalEmbeddingAssistedConfig,
) -> dict[str, Any]:
    """Run v4_1_medical_embedding_assisted and write selector-compatible outputs."""
    start_time = time.perf_counter()
    root_dir = config.root_dir.resolve()
    wsi_path = config.wsi_path.expanduser().resolve()
    output_dir = _resolve_output_dir(config.output_dir, root_dir=root_dir)
    min_distance_level0 = config.min_distance_level0 or config.patch_size
    warnings: list[str] = []

    _validate_config(config=config, wsi_path=wsi_path)
    embedding_cache_path, embedding_cache_metadata_path = _embedding_cache_paths(config, output_dir)
    if (
        config.embedding_model_path is None
        and not (config.reuse_embedding_cache and embedding_cache_path.exists() and embedding_cache_metadata_path.exists())
    ):
        raise RuntimeError(UNI_BACKEND_MISSING_MESSAGE)

    selected_dir = output_dir / "selected"
    candidate_metadata_path = output_dir / "candidate_metadata.csv"
    selected_metadata_path = output_dir / "selected_metadata.csv"
    summary_path = output_dir / "selection_summary.json"
    method_config_path = output_dir / "method_config.json"
    preview_path = output_dir / "patch_selection_preview.png"
    scored_candidates_path = output_dir / SCORED_CANDIDATES_FILE
    cluster_summary_path = output_dir / EMBEDDING_CLUSTER_SUMMARY_FILE

    extractor, tiatoolbox_version = _build_tiatoolbox_extractor(
        wsi_path=wsi_path,
        config=config,
    )
    _prepare_output_dir(
        output_dir=output_dir,
        root_dir=root_dir,
        overwrite=config.overwrite,
    )
    write_json_manifest(
        _method_config(
            config,
            min_distance_level0=min_distance_level0,
            embedding_cache_path=embedding_cache_path,
            tiatoolbox_version=tiatoolbox_version,
        ),
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
        candidates = _candidates_from_extractor(
            extractor=extractor,
            patch_size=config.patch_size,
        )
        num_candidates_generated = len(candidates)
        candidates_to_score = _select_candidates_to_score(
            candidates,
            seed=config.seed,
            max_candidates_to_score=config.max_candidates_to_score,
        )
        candidate_rows = [
            _v41_candidate_pool_row(
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

        scored_records: list[dict[str, object]] = []
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
        _apply_medical_features(
            scored_records,
            slide=slide,
            candidates_to_score=candidates_to_score,
            config=config,
        )
        embeddings, embedding_cache_used, embedding_cache_path, embedding_cache_metadata_path = (
            _load_or_compute_embeddings_v41(
                slide=slide,
                candidates_to_score=candidates_to_score,
                config=config,
                output_dir=output_dir,
                warnings=warnings,
            )
        )
        embedding_stats, embedding_warnings = _apply_embedding_metrics(
            records=scored_records,
            embeddings=embeddings,
            config=config,
        )
        warnings.extend(embedding_warnings)
        _apply_v41_static_scores(scored_records)
        selected_records, selection_stats, selection_warnings = _select_v41_records(
            records=scored_records,
            embeddings=embeddings,
            config=config,
            slide_dimensions=slide_dimensions,
            min_distance_level0=min_distance_level0,
        )
        warnings.extend(selection_warnings)
        warnings.extend(
            _sanitize_numeric_fields(
                scored_records,
                [*V41_CRITICAL_NUMERIC_FIELDS, *V3_NORMALIZED_FIELDS, *MEDICAL_IMAGE_FEATURE_FIELDS],
            )
        )

        embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[1] else None
        for record in scored_records:
            row = candidate_rows_by_id[str(record["candidate_id"])]
            _update_candidate_row_with_v41(
                row,
                record,
                embedding_backend=config.embedding_backend,
                embedding_model_name=config.embedding_model_name,
                embedding_dim=embedding_dim,
                embedding_cache_used=embedding_cache_used,
            )

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
        for row in scored_candidate_rows:
            row["feature_size"] = config.feature_size
        write_csv_manifest(
            rows=scored_candidate_rows,
            output_path=scored_candidates_path,
            fieldnames=SCORED_CANDIDATE_FIELDS,
        )
        _write_cluster_summary(
            records=scored_records,
            output_path=cluster_summary_path,
        )
        save_wsi_patch_selection_preview(
            thumbnail=thumbnail,
            candidate_rows=scored_candidate_rows,
            slide_dimensions=slide_dimensions,
            output_path=preview_path,
        )

        score_fields = [
            "score_v3_base",
            "medical_image_quality_score",
            "medical_image_utility_score",
            "medical_texture_score",
            "medical_pseudo_cellularity_score",
            "medical_artifact_penalty",
            "embedding_representativeness_score",
            "embedding_novelty_score",
            "embedding_diversity_bonus",
            "embedding_redundancy_penalty",
            "morphology_diversity_score",
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
            "version": V41_SELECTOR_VERSION,
            "wsi_path": str(wsi_path),
            "output_dir": str(output_dir),
            "patch_size": config.patch_size,
            "stride": config.stride,
            "max_patches": config.max_patches,
            "min_tissue_ratio": config.min_tissue_ratio,
            "min_mask_ratio": config.min_tissue_ratio,
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
            "medical_min_quality_score": config.medical_min_quality_score,
            "medical_min_utility_score": config.medical_min_utility_score,
            "min_score_v3_base_quantile": config.min_score_v3_base_quantile,
            "medical_top_k_candidates": config.medical_top_k_candidates,
            "medical_top_quantile": config.medical_top_quantile,
            "medical_artifact_max": config.medical_artifact_max,
            "medical_rerank_mode": config.medical_rerank_mode,
            "medical_image_feature_methods": MEDICAL_IMAGE_FEATURE_METHODS,
            "embedding_backend": config.embedding_backend,
            "embedding_model_name": config.embedding_model_name,
            "embedding_dim": embedding_dim,
            "embedding_cache_used": embedding_cache_used,
            "embedding_cache_path": str(embedding_cache_path),
            "embedding_cache_metadata_path": str(embedding_cache_metadata_path),
            "embedding_distance_metric": config.embedding_distance_metric,
            "embedding_cluster_count": config.embedding_cluster_count,
            "selected_clusters": selection_stats.get("selected_clusters", {}),
            "candidate_clusters": selection_stats.get("candidate_clusters", {}),
            "clustering_method": embedding_stats.get("clustering_method"),
            "cache_embeddings": config.cache_embeddings,
            "reuse_embedding_cache": config.reuse_embedding_cache,
            **slide_metadata,
            "num_candidates_generated": num_candidates_generated,
            "num_thumbnail_candidates_passing_mask": len(candidates),
            "num_candidate_rows_written": len(candidate_rows),
            "num_candidates_scored": len(scored_records),
            "num_candidates_evaluated": len(scored_records),
            "num_selected": len(selected_rows),
            "score_statistics": score_statistics,
            "selected_category_counts": selection_stats.get("selected_category_counts", {}),
            "spatial_coverage": {
                "regions_covered": selection_stats.get("regions_covered"),
                "active_regions": selection_stats.get("active_regions"),
                "patches_per_region": selection_stats.get("patches_per_region"),
                "quota_fill_rate": selection_stats.get("quota_fill_rate"),
            },
            "quality_eligible_candidates": selection_stats.get("eligible_candidates"),
            "v3_top_candidates": selection_stats.get("v3_top_candidates"),
            "medical_top_candidates": selection_stats.get("medical_top_candidates"),
            "strict_eligible_candidates": selection_stats.get("strict_eligible_candidates"),
            "relaxed_eligible_candidates": selection_stats.get("relaxed_eligible_candidates"),
            "eligibility_stage": selection_stats.get("eligibility_stage"),
            "v3_base_threshold": selection_stats.get("v3_base_threshold"),
            "mean_score_v3_base_selected": _selected_mean(selected_rows, "score_v3_base"),
            "mean_medical_image_quality_score_selected": _selected_mean(
                selected_rows,
                "medical_image_quality_score",
            ),
            "mean_medical_image_utility_score_selected": _selected_mean(
                selected_rows,
                "medical_image_utility_score",
            ),
            "mean_medical_artifact_penalty_selected": _selected_mean(
                selected_rows,
                "medical_artifact_penalty",
            ),
            "mean_morphology_diversity_score_selected": _selected_mean(
                selected_rows,
                "morphology_diversity_score",
            ),
            "mean_score_final_selected": _selected_mean(selected_rows, "score_final"),
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "candidate_metadata_csv": str(candidate_metadata_path),
            "selected_metadata_csv": str(selected_metadata_path),
            "scored_candidates_csv": str(scored_candidates_path),
            "embedding_cluster_summary_csv": str(cluster_summary_path),
            "method_config_json": str(method_config_path),
            "preview_image": str(preview_path),
            "selected_dir": str(selected_dir),
            "candidate_pool": TIATOOLBOX_CANDIDATE_POOL,
            "candidate_metadata_semantics": TIATOOLBOX_CANDIDATE_METADATA_SEMANTICS,
            "candidate_pool_definition": (
                "TIAToolbox sliding-window candidates passing Otsu input mask and min_mask_ratio."
            ),
            "candidate_pool_shared_with_baseline": True,
            "shared_candidate_pool_selector": "baseline_tiatoolbox",
            "preview_shows": V41_PREVIEW_SHOWS,
            "candidate_ordering": V41_CANDIDATE_ORDERING,
            "extraction_backend": TIATOOLBOX_EXTRACTION_BACKEND,
            "tiatoolbox_version": tiatoolbox_version,
            "tiatoolbox_extractor": TIATOOLBOX_EXTRACTOR_NAME,
            "input_mask": TIATOOLBOX_INPUT_MASK,
            "tissue_mask_method": TIATOOLBOX_TISSUE_MASK_METHOD,
            "nuclear_signal_rgb_method": RGB_NUCLEAR_SIGNAL_METHOD,
            "nuclear_signal_hed_method": HED_NUCLEAR_SIGNAL_METHOD,
            "visual_entropy_method": VISUAL_ENTROPY_METHOD,
            "blur_score_method": BLUR_SCORE_METHOD,
            "artifact_penalty_method": ARTIFACT_PENALTY_METHOD,
            "v3_weights": V3_WEIGHTS,
            "v4_1_weights": {
                **V41_WEIGHTS,
                "embedding_diversity": config.embedding_diversity_weight,
                "embedding_redundancy": config.embedding_redundancy_weight,
                "cluster_balance": config.cluster_balance_weight,
                "representative_cluster": config.representative_cluster_weight,
            },
            "useful_patch_definition": USEFUL_PATCH_DEFINITION,
            "embedding_selection_note": EMBEDDING_SELECTION_NOTE,
            "no_deep_learning_used_for_selection": False,
            "segmentation_model_used_for_selection": False,
            "selection_model_note": NO_MODEL_SELECTION_NOTE,
            "clinical_warning": CLINICAL_WARNING,
            "warnings": warnings,
        }
        write_json_manifest(summary, summary_path)
        return summary
    finally:
        slide.close()
