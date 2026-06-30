"""Normalización de features para el scoring técnico."""

from __future__ import annotations

import math
from typing import Iterable


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_feature(values: Iterable[object]) -> list[float]:
    """Normaliza a [0, 1] y usa 0.5 para series constantes o valores inválidos."""
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
