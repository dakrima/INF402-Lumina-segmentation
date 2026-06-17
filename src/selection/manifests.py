"""Manifest writers for patch selection outputs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATE_METADATA_FIELDS = [
    "candidate_id",
    "grid_index",
    "x_level0",
    "y_level0",
    "patch_size",
    "width",
    "height",
    "thumbnail_tissue_ratio",
    "evaluated",
    "scored",
    "nuclear_proxy",
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
    "score_v3_base",
    "embedding_backend",
    "embedding_model_name",
    "embedding_dim",
    "embedding_cache_used",
    "embedding_cluster_id",
    "embedding_distance_to_cluster_centroid",
    "embedding_representativeness_score",
    "embedding_novelty_score",
    "embedding_diversity_bonus",
    "embedding_redundancy_penalty",
    "morphology_diversity_score",
    "score_raw",
    "score_final",
    "usefulness_reason",
    "region_id",
    "region_row",
    "region_col",
    "quota_grid",
    "spatial_strategy",
    "diversity_strategy",
    "selected",
    "rank",
    "filename",
    "patch_id",
    "patch_path",
    "selector",
    "selection_method",
    "seed",
    "source_wsi_path",
    "slide_width",
    "slide_height",
    "objective_power",
    "mpp_x",
    "mpp_y",
    "level_count",
]

SELECTED_METADATA_FIELDS = [
    "patch_id",
    "filename",
    "selected",
    "rank",
    "x_level0",
    "y_level0",
    "patch_size",
    "width",
    "height",
    "thumbnail_tissue_ratio",
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
    "score_v3_base",
    "embedding_backend",
    "embedding_model_name",
    "embedding_dim",
    "embedding_cache_used",
    "embedding_cluster_id",
    "embedding_distance_to_cluster_centroid",
    "embedding_representativeness_score",
    "embedding_novelty_score",
    "embedding_diversity_bonus",
    "embedding_redundancy_penalty",
    "morphology_diversity_score",
    "score_raw",
    "score_final",
    "usefulness_reason",
    "nuclear_proxy",
    "region_id",
    "region_row",
    "region_col",
    "quota_grid",
    "spatial_strategy",
    "diversity_strategy",
    "patch_path",
    "selector",
    "source_wsi_path",
    "slide_width",
    "slide_height",
    "objective_power",
    "mpp_x",
    "mpp_y",
    "level_count",
    "selection_method",
    "seed",
]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_csv_manifest(
    rows: list[dict[str, object]],
    output_path: Path,
    fieldnames: list[str],
) -> Path:
    """Write rows to CSV with stable columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_json_manifest(payload: dict[str, Any], output_path: Path) -> Path:
    """Write an indented JSON manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
