"""Scoring técnico reutilizado por el selector propuesto."""

from __future__ import annotations

import math
import random
from typing import Any

from src.selection.quality_filters import compute_patch_features
from src.selection.scoring import normalize_feature


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


def format_float(value: object) -> str:
    """Formatea un valor numérico con la precisión usada en los manifiestos."""
    return f"{float(value):.6f}"


def safe_float(value: object, default: float = 0.0) -> float:
    """Convierte un valor a `float` finito o retorna el valor por defecto."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def select_candidates_to_score(
    candidates: list[Any],
    *,
    seed: int,
    max_candidates_to_score: int,
) -> list[Any]:
    """
    ***
    * candidates: Pool común generado por TIAToolbox.
    * seed: Semilla que fija el orden reproducible.
    * max_candidates_to_score: Límite de candidatos; cero conserva todo el pool.
    ***
    Mezcla una copia del pool con la semilla del experimento y retorna el subconjunto
    que recibirá scoring técnico.
    """
    ordered_candidates = list(candidates)
    random.Random(seed).shuffle(ordered_candidates)
    return ordered_candidates if max_candidates_to_score == 0 else ordered_candidates[:max_candidates_to_score]


def _compute_v3_raw_features(patch_image: object, *, feature_size: int) -> dict[str, float]:
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
    residual = safe_float(record.get("residual_candidate_proxy"))
    treated = safe_float(record.get("low_cellularity_treated_bed_proxy"))
    heterogeneity = safe_float(record.get("heterogeneity_score"))
    quality = safe_float(record.get("technical_quality_score"))

    if treated >= max(residual, heterogeneity, quality):
        return "low_cellularity_treated_bed_proxy"
    if residual >= max(treated, heterogeneity, quality):
        return "higher_cellularity_residual_candidate_proxy"
    if heterogeneity >= max(treated, residual, quality):
        return "heterogeneous_region_proxy"
    return "technical_quality_balanced_proxy"


def apply_v3_scores(records: list[dict[str, object]]) -> None:
    """
    ***
    * records: Características clásicas de cada patch candidato.
    ***
    Normaliza las características y aplica las fórmulas técnicas originales de v3.
    Modifica los registros en el mismo orden y no cambia los pesos del experimento.
    """
    normalized_by_feature = {
        "tissue_ratio": normalize_feature(record.get("tissue_ratio") for record in records),
        "nuclear_signal_rgb": normalize_feature(record.get("nuclear_signal_rgb") for record in records),
        "nuclear_signal_hed": normalize_feature(record.get("nuclear_signal_hed") for record in records),
        "visual_entropy": normalize_feature(record.get("visual_entropy") for record in records),
        "blur_score": normalize_feature(record.get("blur_score") for record in records),
        "artifact_penalty": normalize_feature(record.get("artifact_penalty") for record in records),
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
            0.45 * cellularity_proxy + 0.30 * heterogeneity + 0.25 * technical_quality
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

        record.update(
            {
                "tissue_ratio_norm": tissue_norm,
                "nuclear_signal_rgb_norm": rgb_norm,
                "nuclear_signal_hed_norm": hed_norm,
                "visual_entropy_norm": entropy_norm,
                "blur_score_norm": blur_norm,
                "artifact_penalty_norm": artifact_norm,
                "thumbnail_tissue_ratio_norm": thumbnail_norm,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "technical_quality_score": technical_quality,
                "heterogeneity_score": heterogeneity,
                "cellularity_proxy_score": cellularity_proxy,
                "residual_candidate_proxy": residual_candidate,
                "low_cellularity_treated_bed_proxy": low_cellularity_treated_bed,
                "tumor_bed_relevance_proxy": tumor_bed_relevance,
                "usefulness_score": usefulness,
                "score_raw": usefulness,
            }
        )
        record["usefulness_reason"] = _usefulness_reason(record)


def candidate_record_from_patch(
    candidate: object,
    *,
    patch_image: object,
    config: Any,
) -> dict[str, object]:
    """Construye el registro técnico de un candidato a partir de su imagen RGB."""
    features = _compute_v3_raw_features(patch_image, feature_size=config.feature_size)
    return {
        "candidate_id": candidate.candidate_id,
        "grid_index": candidate.grid_index,
        "x_level0": candidate.x_level0,
        "y_level0": candidate.y_level0,
        "patch_size": candidate.patch_size,
        "width": patch_image.width,
        "height": patch_image.height,
        "thumbnail_tissue_ratio": getattr(candidate, "thumbnail_tissue_ratio", ""),
        "tiatoolbox_index": getattr(candidate, "tiatoolbox_index", ""),
        "feature_size": config.feature_size,
        **features,
    }


def update_candidate_row_from_record(row: dict[str, object], record: dict[str, object]) -> None:
    """Copia el scoring calculado al manifiesto del pool común."""
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
        row[field_name] = format_float(value) if isinstance(value, float) else value
    row["region_id"] = record.get("region_id", "")
    row["region_row"] = record.get("region_row", "")
    row["region_col"] = record.get("region_col", "")
    row["quota_grid"] = record.get("quota_grid", row.get("quota_grid", ""))
    row["selected"] = bool(record.get("selected", False))
    row["rank"] = record.get("rank", "")


def sanitize_numeric_fields(
    records: list[dict[str, object]],
    field_names: list[str],
) -> list[str]:
    """Reemplaza valores críticos no finitos por cero e informa cada reparación."""
    warnings: list[str] = []
    for field_name in field_names:
        repaired_count = 0
        for record in records:
            try:
                number = float(record.get(field_name))
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number):
                record[field_name] = 0.0
                repaired_count += 1
        if repaired_count:
            warnings.append(
                f"Se reemplazaron {repaired_count} valores no finitos de {field_name} por 0.0."
            )
    return warnings


def score_statistics(records: list[dict[str, object]], field_name: str) -> dict[str, float | None]:
    """Resume media, mínimo y máximo de un score finito."""
    values = [safe_float(record.get(field_name), math.nan) for record in records]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": float(sum(values) / len(values)), "min": float(min(values)), "max": float(max(values))}


def selected_mean(selected_rows: list[dict[str, object]], field_name: str) -> float | None:
    """Calcula la media finita de un campo en los patches seleccionados."""
    values = [safe_float(row.get(field_name), math.nan) for row in selected_rows]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else None
