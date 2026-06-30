"""Helpers de embeddings y clustering usados por el selector propuesto."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.selection.embedding_scoring import (
    EmbeddingExtractorConfig,
    build_embedding_extractor,
    cluster_embeddings,
    compute_patch_embeddings,
    embedding_cluster_metrics,
)
from src.selection.manifests import write_csv_manifest
from src.selection.technical_scoring import safe_float


EMBEDDING_CACHE_FILE = "embedding_cache.npz"
EMBEDDING_CACHE_METADATA_FILE = "embedding_cache_metadata.json"
EMBEDDING_CLUSTER_SUMMARY_FILE = "embedding_cluster_summary.csv"
SCORED_CANDIDATES_FILE = "scored_candidates.csv"

V4_EMBEDDING_FIELDS = [
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
]

EMBEDDING_CLUSTER_SUMMARY_FIELDS = [
    "embedding_cluster_id",
    "num_candidates",
    "num_selected",
    "mean_score_v3_base",
    "mean_embedding_representativeness_score",
]


def embedding_cache_paths(config: Any, output_dir: Path) -> tuple[Path, Path]:
    """Resuelve las rutas del cache UNI sin escribir archivos."""
    if config.embedding_cache_path is None:
        return output_dir / EMBEDDING_CACHE_FILE, output_dir / EMBEDDING_CACHE_METADATA_FILE
    cache_path = config.embedding_cache_path.expanduser()
    if not cache_path.is_absolute():
        cache_path = output_dir / cache_path
    cache_path = cache_path.resolve()
    return cache_path, cache_path.with_name(f"{cache_path.stem}_metadata.json")


def compute_embeddings_for_candidates(
    *,
    slide: object,
    candidates: list[object],
    config: Any,
    embedding_extractor: object | None = None,
) -> np.ndarray:
    """
    ***
    * slide: WSI abierta con OpenSlide.
    * candidates: Candidatos con coordenadas en nivel 0.
    * config: Configuración inmutable del selector propuesto.
    * embedding_extractor: Instancia UNI compartida, cuando ya fue cargada.
    ***
    Lee los patches en lotes y genera sus embeddings UNI en el mismo orden.
    Retorna una matriz `num_patches x embedding_dim`.
    """
    extractor = embedding_extractor or build_embedding_extractor(
        EmbeddingExtractorConfig(
            embedding_backend=config.embedding_backend,
            embedding_model_name=config.embedding_model_name,
            embedding_model_path=config.embedding_model_path,
            embedding_device=config.embedding_device,
            embedding_batch_size=config.embedding_batch_size,
            embedding_num_workers=config.embedding_num_workers,
            embedding_dim=config.embedding_dim,
            embedding_distance_metric=config.embedding_distance_metric,
        )
    )
    batches: list[np.ndarray] = []
    for start in range(0, len(candidates), config.embedding_batch_size):
        batch_candidates = candidates[start:start + config.embedding_batch_size]
        patches = [
            slide.read_region(
                (candidate.x_level0, candidate.y_level0),
                0,
                (config.patch_size, config.patch_size),
            ).convert("RGB")
            for candidate in batch_candidates
        ]
        batches.append(compute_patch_embeddings(extractor, patches))
        del patches
    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if config.embedding_dim is not None and embeddings.shape[1] != config.embedding_dim:
        raise RuntimeError(
            f"Dimensión de embedding inválida: se esperaba {config.embedding_dim} y se obtuvo {embeddings.shape[1]}."
        )
    return embeddings


def apply_embedding_metrics(
    *,
    records: list[dict[str, object]],
    embeddings: np.ndarray,
    config: Any,
) -> tuple[dict[str, object], list[str]]:
    """Agrupa los embeddings y agrega métricas morfológicas a cada candidato."""
    labels, centroids, clustering_method, warnings = cluster_embeddings(
        embeddings,
        cluster_count=config.embedding_cluster_count,
        seed=config.seed,
        distance_metric=config.embedding_distance_metric,
    )
    distances, representative_scores = embedding_cluster_metrics(
        embeddings,
        labels,
        centroids,
        distance_metric=config.embedding_distance_metric,
    )
    for index, record in enumerate(records):
        record.update(
            {
                "score_v3_base": float(record["score_raw"]),
                "embedding_backend": config.embedding_backend,
                "embedding_model_name": config.embedding_model_name,
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "embedding_cluster_id": int(labels[index]),
                "embedding_distance_to_cluster_centroid": float(distances[index]),
                "embedding_representativeness_score": float(representative_scores[index]),
                "embedding_novelty_score": 0.0,
                "embedding_diversity_bonus": 0.0,
                "embedding_redundancy_penalty": 0.0,
                "morphology_diversity_score": float(representative_scores[index]),
            }
        )
    cluster_counts = Counter(int(label) for label in labels)
    return {
        "clustering_method": clustering_method,
        "cluster_count": len(cluster_counts),
        "candidate_clusters": dict(sorted((str(key), value) for key, value in cluster_counts.items())),
    }, warnings


def write_cluster_summary(*, records: list[dict[str, object]], output_path: Path) -> Path:
    """Guarda el resumen por cluster del reranking morfológico."""
    rows: list[dict[str, object]] = []
    cluster_ids = sorted({str(record.get("embedding_cluster_id", "")) for record in records})
    for cluster_id in cluster_ids:
        cluster_records = [
            record for record in records
            if str(record.get("embedding_cluster_id", "")) == cluster_id
        ]
        selected_records = [
            record for record in cluster_records
            if record.get("selected") in (True, "True", "true", "1")
        ]
        rows.append(
            {
                "embedding_cluster_id": cluster_id,
                "num_candidates": len(cluster_records),
                "num_selected": len(selected_records),
                "mean_score_v3_base": _mean_record_value(cluster_records, "score_v3_base"),
                "mean_embedding_representativeness_score": _mean_record_value(
                    cluster_records,
                    "embedding_representativeness_score",
                ),
            }
        )
    return write_csv_manifest(
        rows=rows,
        output_path=output_path,
        fieldnames=EMBEDDING_CLUSTER_SUMMARY_FIELDS,
    )


def _mean_record_value(records: list[dict[str, object]], field_name: str) -> float | None:
    values = [safe_float(record.get(field_name), float("nan")) for record in records]
    values = [value for value in values if np.isfinite(value)]
    return float(sum(values) / len(values)) if values else None
