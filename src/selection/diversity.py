"""Spatial diversity utilities for greedy patch selection."""

from __future__ import annotations

import math
import statistics


def patch_center(record: dict[str, object]) -> tuple[float, float]:
    patch_size = float(record["patch_size"])
    return (
        float(record["x_level0"]) + patch_size / 2.0,
        float(record["y_level0"]) + patch_size / 2.0,
    )


def proximity_penalty(
    candidate: dict[str, object],
    selected_records: list[dict[str, object]],
    min_distance_level0: float,
) -> float:
    """Return max exponential proximity penalty against selected patch centers."""
    if not selected_records:
        return 0.0
    if min_distance_level0 <= 0:
        return 0.0

    cx, cy = patch_center(candidate)
    penalties: list[float] = []
    for selected in selected_records:
        if selected.get("candidate_id") == candidate.get("candidate_id"):
            continue
        sx, sy = patch_center(selected)
        distance = math.hypot(cx - sx, cy - sy)
        penalties.append(math.exp(-distance / min_distance_level0))
    if not penalties:
        return 0.0
    return float(max(penalties))


def parse_quota_grid(quota_grid: str) -> tuple[int, int]:
    """Parse a quota grid string like 4x4."""
    normalized = quota_grid.lower().strip()
    if "x" not in normalized:
        raise ValueError("--quota-grid must use format ROWSxCOLS, for example 4x4.")
    rows_text, cols_text = normalized.split("x", 1)
    rows = int(rows_text)
    cols = int(cols_text)
    if rows <= 0 or cols <= 0:
        raise ValueError("--quota-grid rows and columns must be positive.")
    return rows, cols


def assign_spatial_regions(
    records: list[dict[str, object]],
    slide_dimensions: tuple[int, int],
    quota_grid: str,
) -> None:
    """Mutate records with region_id, region_row, region_col and quota_grid."""
    grid_rows, grid_cols = parse_quota_grid(quota_grid)
    slide_width, slide_height = slide_dimensions
    for record in records:
        center_x, center_y = patch_center(record)
        region_col = min(grid_cols - 1, max(0, int(center_x / slide_width * grid_cols)))
        region_row = min(grid_rows - 1, max(0, int(center_y / slide_height * grid_rows)))
        record["region_row"] = region_row
        record["region_col"] = region_col
        record["region_id"] = f"r{region_row}_c{region_col}"
        record["quota_grid"] = quota_grid


def feature_diversity_bonus(
    candidate: dict[str, object],
    selected_records: list[dict[str, object]],
    feature_names: list[str],
) -> float:
    """Return min feature-space distance to selected records."""
    if not selected_records:
        return 0.0
    candidate_vector = [float(candidate.get(feature_name, 0.0)) for feature_name in feature_names]
    distances: list[float] = []
    for selected in selected_records:
        selected_vector = [float(selected.get(feature_name, 0.0)) for feature_name in feature_names]
        squared = [
            (candidate_value - selected_value) ** 2
            for candidate_value, selected_value in zip(candidate_vector, selected_vector)
        ]
        distances.append(math.sqrt(sum(squared) / len(squared)) if squared else 0.0)
    return float(min(distances)) if distances else 0.0


def _score_with_penalties(
    record: dict[str, object],
    selected_records: list[dict[str, object]],
    *,
    lambda_spatial: float,
    min_distance_level0: float,
    diversity_strategy: str,
    feature_diversity_weight: float,
    feature_names: list[str],
) -> tuple[float, float, float]:
    spatial_penalty = proximity_penalty(
        candidate=record,
        selected_records=selected_records,
        min_distance_level0=min_distance_level0,
    )
    diversity_bonus = (
        feature_diversity_bonus(
            candidate=record,
            selected_records=selected_records,
            feature_names=feature_names,
        )
        if diversity_strategy == "farthest_feature"
        else 0.0
    )
    final_score = (
        float(record["score_raw"])
        - lambda_spatial * spatial_penalty
        + feature_diversity_weight * diversity_bonus
    )
    return float(final_score), float(spatial_penalty), float(diversity_bonus)


def greedy_select_with_spatial_penalty(
    records: list[dict[str, object]],
    max_patches: int,
    lambda_spatial: float,
    min_distance_level0: float,
    diversity_strategy: str = "none",
    feature_diversity_weight: float = 0.0,
    feature_names: list[str] | None = None,
) -> list[dict[str, object]]:
    """Select records by score_raw with a dynamic spatial redundancy penalty."""
    available = list(records)
    selected: list[dict[str, object]] = []
    feature_names = feature_names or []

    while available and len(selected) < max_patches:
        best_record: dict[str, object] | None = None
        best_score = -math.inf
        best_penalty = 0.0
        best_diversity_bonus = 0.0

        for record in available:
            final_score, penalty, diversity_bonus = _score_with_penalties(
                record,
                selected,
                lambda_spatial=lambda_spatial,
                min_distance_level0=min_distance_level0,
                diversity_strategy=diversity_strategy,
                feature_diversity_weight=feature_diversity_weight,
                feature_names=feature_names,
            )
            tie_breaker = str(record.get("candidate_id", ""))
            best_tie_breaker = str(best_record.get("candidate_id", "")) if best_record else ""
            if final_score > best_score or (
                math.isclose(final_score, best_score) and tie_breaker < best_tie_breaker
            ):
                best_record = record
                best_score = final_score
                best_penalty = penalty
                best_diversity_bonus = diversity_bonus

        if best_record is None:
            break

        best_record["selected"] = True
        best_record["rank"] = len(selected) + 1
        best_record["spatial_penalty"] = float(best_penalty)
        best_record["feature_diversity_bonus"] = float(best_diversity_bonus)
        best_record["score_final"] = float(best_score)
        selected.append(best_record)
        available.remove(best_record)

    for record in available:
        final_score, penalty, diversity_bonus = _score_with_penalties(
            record,
            selected,
            lambda_spatial=lambda_spatial,
            min_distance_level0=min_distance_level0,
            diversity_strategy=diversity_strategy,
            feature_diversity_weight=feature_diversity_weight,
            feature_names=feature_names,
        )
        record["selected"] = False
        record["rank"] = ""
        record["spatial_penalty"] = float(penalty)
        record["feature_diversity_bonus"] = float(diversity_bonus)
        record["score_final"] = float(final_score)

    return selected


def score_quantile(records: list[dict[str, object]], quantile: float) -> float:
    """Return a nearest-rank score_raw quantile for soft quota filtering."""
    if not records:
        return 0.0
    clipped_quantile = max(0.0, min(1.0, quantile))
    scores = sorted(float(record["score_raw"]) for record in records)
    index = min(len(scores) - 1, max(0, int(round((len(scores) - 1) * clipped_quantile))))
    return float(scores[index])


def select_with_spatial_quotas(
    records: list[dict[str, object]],
    max_patches: int,
    lambda_spatial: float,
    min_distance_level0: float,
    quota_min_score_quantile: float,
    diversity_strategy: str,
    feature_diversity_weight: float,
    feature_names: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select records using soft regional quotas and automatic quota redistribution."""
    threshold = score_quantile(records, quota_min_score_quantile)
    eligible = [
        record for record in records
        if float(record["score_raw"]) >= threshold
    ]
    regions = sorted({str(record.get("region_id", "")) for record in eligible if record.get("region_id")})
    selected: list[dict[str, object]] = []
    available = list(eligible)
    region_index = 0

    while available and len(selected) < max_patches and regions:
        made_selection = False
        for _ in range(len(regions)):
            region_id = regions[region_index % len(regions)]
            region_index += 1
            candidates = [
                record for record in available
                if record.get("region_id") == region_id
            ]
            if not candidates:
                continue

            best_record: dict[str, object] | None = None
            best_score = -math.inf
            best_penalty = 0.0
            best_diversity_bonus = 0.0
            for record in candidates:
                final_score, penalty, diversity_bonus = _score_with_penalties(
                    record,
                    selected,
                    lambda_spatial=lambda_spatial,
                    min_distance_level0=min_distance_level0,
                    diversity_strategy=diversity_strategy,
                    feature_diversity_weight=feature_diversity_weight,
                    feature_names=feature_names,
                )
                tie_breaker = str(record.get("candidate_id", ""))
                best_tie_breaker = str(best_record.get("candidate_id", "")) if best_record else ""
                if final_score > best_score or (
                    math.isclose(final_score, best_score) and tie_breaker < best_tie_breaker
                ):
                    best_record = record
                    best_score = final_score
                    best_penalty = penalty
                    best_diversity_bonus = diversity_bonus

            if best_record is None:
                continue

            best_record["selected"] = True
            best_record["rank"] = len(selected) + 1
            best_record["spatial_penalty"] = float(best_penalty)
            best_record["feature_diversity_bonus"] = float(best_diversity_bonus)
            best_record["score_final"] = float(best_score)
            selected.append(best_record)
            available.remove(best_record)
            made_selection = True
            break

        if not made_selection:
            break

    selected_ids = {record["candidate_id"] for record in selected}
    for record in records:
        if record["candidate_id"] in selected_ids:
            continue
        final_score, penalty, diversity_bonus = _score_with_penalties(
            record,
            selected,
            lambda_spatial=lambda_spatial,
            min_distance_level0=min_distance_level0,
            diversity_strategy=diversity_strategy,
            feature_diversity_weight=feature_diversity_weight,
            feature_names=feature_names,
        )
        record["selected"] = False
        record["rank"] = ""
        record["spatial_penalty"] = float(penalty)
        record["feature_diversity_bonus"] = float(diversity_bonus)
        record["score_final"] = float(final_score)

    patches_per_region: dict[str, int] = {}
    for record in selected:
        region_id = str(record.get("region_id", ""))
        patches_per_region[region_id] = patches_per_region.get(region_id, 0) + 1

    active_region_ids = {
        str(record.get("region_id", ""))
        for record in records
        if record.get("region_id")
    }
    selected_region_ids = set(patches_per_region)
    quota_stats = {
        "score_threshold": threshold,
        "active_regions": len(active_region_ids),
        "regions_covered": len(selected_region_ids),
        "patches_per_region": dict(sorted(patches_per_region.items())),
        "quota_fill_rate": len(selected_region_ids) / len(active_region_ids)
        if active_region_ids
        else None,
        "quota_min_score_quantile": quota_min_score_quantile,
        "eligible_candidates": len(eligible),
        "mean_feature_diversity_bonus_selected": (
            float(statistics.mean(float(record.get("feature_diversity_bonus", 0.0)) for record in selected))
            if selected
            else None
        ),
    }
    return selected, quota_stats
