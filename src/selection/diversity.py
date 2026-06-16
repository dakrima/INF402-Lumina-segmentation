"""Spatial diversity utilities for greedy patch selection."""

from __future__ import annotations

import math


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


def greedy_select_with_spatial_penalty(
    records: list[dict[str, object]],
    max_patches: int,
    lambda_spatial: float,
    min_distance_level0: float,
) -> list[dict[str, object]]:
    """Select records by score_raw with a dynamic spatial redundancy penalty."""
    available = list(records)
    selected: list[dict[str, object]] = []

    while available and len(selected) < max_patches:
        best_record: dict[str, object] | None = None
        best_score = -math.inf
        best_penalty = 0.0

        for record in available:
            penalty = proximity_penalty(
                candidate=record,
                selected_records=selected,
                min_distance_level0=min_distance_level0,
            )
            final_score = float(record["score_raw"]) - lambda_spatial * penalty
            tie_breaker = str(record.get("candidate_id", ""))
            best_tie_breaker = str(best_record.get("candidate_id", "")) if best_record else ""
            if final_score > best_score or (
                math.isclose(final_score, best_score) and tie_breaker < best_tie_breaker
            ):
                best_record = record
                best_score = final_score
                best_penalty = penalty

        if best_record is None:
            break

        best_record["selected"] = True
        best_record["rank"] = len(selected) + 1
        best_record["spatial_penalty"] = float(best_penalty)
        best_record["score_final"] = float(best_score)
        selected.append(best_record)
        available.remove(best_record)

    for record in available:
        penalty = proximity_penalty(
            candidate=record,
            selected_records=selected,
            min_distance_level0=min_distance_level0,
        )
        record["selected"] = False
        record["rank"] = ""
        record["spatial_penalty"] = float(penalty)
        record["score_final"] = float(float(record["score_raw"]) - lambda_spatial * penalty)

    return selected
