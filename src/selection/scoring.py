"""Feature normalization and scoring for smart patch selection."""

from __future__ import annotations

import math
from typing import Iterable


DEFAULT_SMART_WEIGHTS = {
    "tissue_ratio": 0.35,
    "nuclear_signal": 0.25,
    "visual_entropy": 0.15,
    "blur_score": 0.15,
    "artifact_penalty": -0.10,
}


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_feature(values: Iterable[object]) -> list[float]:
    """Normalize values to [0, 1], returning 0.5 for constant or invalid series."""
    raw_values = list(values)
    safe_values = [_safe_float(value) for value in raw_values]
    valid_values = [value for value in safe_values if value is not None]
    if not valid_values:
        return [0.5 for _ in raw_values]

    minimum = min(valid_values)
    maximum = max(valid_values)
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return [0.5 for _ in raw_values]

    scale = maximum - minimum
    normalized: list[float] = []
    for value in safe_values:
        if value is None:
            normalized.append(0.5)
        else:
            normalized.append(max(0.0, min(1.0, (value - minimum) / scale)))
    return normalized


def apply_feature_scores(
    records: list[dict[str, object]],
    weights: dict[str, float] | None = None,
) -> None:
    """Mutate scored records with normalized features and score_raw."""
    score_weights = weights or DEFAULT_SMART_WEIGHTS
    normalized_by_feature = {
        feature_name: normalize_feature(record.get(feature_name) for record in records)
        for feature_name in score_weights
    }

    for index, record in enumerate(records):
        score_raw = 0.0
        for feature_name, weight in score_weights.items():
            normalized_value = normalized_by_feature[feature_name][index]
            record[f"{feature_name}_norm"] = normalized_value
            score_raw += weight * normalized_value
        record["score_raw"] = float(score_raw)
