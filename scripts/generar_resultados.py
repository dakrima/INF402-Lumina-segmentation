#!/usr/bin/env python
"""Genera las métricas morfológicas finales desde los embeddings UNI persistidos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.selection.embedding_scoring import load_embedding_cache, normalize_embeddings
from src.selection.manifests import write_csv_manifest, write_json_manifest


DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "results/runs/inf402_n9"
METHODS = {
    "baseline": "baseline_tiatoolbox",
    "v4_1": "v4_1_medical_embedding_assisted",
}
PER_WSI_FIELDS = [
    "case_id",
    "method",
    "num_patches_selected",
    "num_valid_embeddings",
    "embedding_dim",
    "mean_pairwise_cosine_distance",
    "sd_pairwise_cosine_distance",
    "median_pairwise_cosine_distance",
    "min_pairwise_cosine_distance",
    "max_pairwise_cosine_distance",
    "mean_nearest_neighbor_cosine_distance",
    "median_nearest_neighbor_cosine_distance",
    "delta_mean_pairwise_v41_minus_baseline",
    "delta_mean_nearest_neighbor_v41_minus_baseline",
    "embedding_references_reused",
    "embeddings_recomputed",
    "embedding_configuration_hash",
    "selected_patch_fingerprint",
    "status",
]
AGGREGATE_FIELDS = [
    "row_type",
    "metric",
    "method",
    "n",
    "mean",
    "sd",
    "median",
    "q1",
    "q3",
    "iqr",
    "v41_greater_wsi",
    "baseline_greater_wsi",
    "ties",
]
METRICS = {
    "mean_pairwise_cosine_distance": "Distancia coseno media entre pares",
    "mean_nearest_neighbor_cosine_distance": "Distancia coseno media al vecino más cercano",
}
CONFIG_FIELDS = (
    "embedding_backend",
    "embedding_model_name",
    "embedding_model_path",
    "embedding_distance_metric",
    "embedding_device",
    "feature_size",
    "patch_size",
)
TOLERANCE = 2e-6


def parse_args() -> argparse.Namespace:
    """
    Construye el parser para localizar una corrida y validar su número de casos.

    Retorna el namespace con la carpeta del experimento, cantidad esperada y opción
    de autocomprobación.
    """
    parser = argparse.ArgumentParser(
        description="Calcula diversidad morfológica con los embeddings UNI persistidos.",
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--expected-cases", type=int, default=9)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lee un manifiesto CSV y retorna sus filas como diccionarios de texto."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def canonical_hash(payload: object) -> str:
    """Calcula SHA-256 sobre una serialización JSON canónica y reproducible."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def descriptive(values: list[float]) -> dict[str, float | int]:
    """
    ***
    * values: Valores de una métrica para los casos evaluados.
    ***
    Calcula cantidad, media, desviación estándar muestral, mediana y cuartiles.

    Retorna un diccionario con los estadísticos descriptivos del paper.
    """
    array = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(array, [25, 75])
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def distance_metrics(embeddings: np.ndarray) -> tuple[dict[str, float], dict[str, bool]]:
    """
    ***
    * embeddings: Matriz con un embedding por patch seleccionado.
    ***
    Normaliza por L2 y calcula distancias coseno entre pares y al vecino más cercano.
    Retorna las métricas y las comprobaciones de validez de la matriz.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("Se requieren al menos dos embeddings bidimensionales.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Los embeddings contienen valores NaN o infinitos.")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Los embeddings contienen vectores nulos.")

    normalized = normalize_embeddings(embeddings)
    distances = 1.0 - normalized @ normalized.T
    symmetric = bool(np.allclose(distances, distances.T, atol=TOLERANCE, rtol=0.0))
    diagonal_zero = bool(np.allclose(np.diag(distances), 0.0, atol=TOLERANCE, rtol=0.0))
    bounds_valid = bool(
        np.isfinite(distances).all()
        and float(np.min(distances)) >= -TOLERANCE
        and float(np.max(distances)) <= 2.0 + TOLERANCE
    )
    if not symmetric or not diagonal_zero or not bounds_valid:
        raise ValueError(
            "Matriz de distancia coseno inválida: "
            f"simétrica={symmetric}, diagonal_cero={diagonal_zero}, límites_válidos={bounds_valid}."
        )

    distances = np.clip(distances, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    nearest = np.min(np.where(np.eye(distances.shape[0], dtype=bool), np.inf, distances), axis=1)
    return {
        "mean_pairwise_cosine_distance": float(np.mean(upper)),
        "sd_pairwise_cosine_distance": float(np.std(upper, ddof=1)),
        "median_pairwise_cosine_distance": float(np.median(upper)),
        "min_pairwise_cosine_distance": float(np.min(upper)),
        "max_pairwise_cosine_distance": float(np.max(upper)),
        "mean_nearest_neighbor_cosine_distance": float(np.mean(nearest)),
        "median_nearest_neighbor_cosine_distance": float(np.median(nearest)),
    }, {
        "symmetric": symmetric,
        "diagonal_zero": diagonal_zero,
        "bounds_valid": bounds_valid,
    }


def selected_candidate_ids(
    case_id: str,
    selected_rows: list[dict[str, str]],
    shared_rows: list[dict[str, str]],
) -> tuple[list[str], str]:
    """
    ***
    * case_id: Identificador persistido de la WSI.
    * selected_rows: Patches seleccionados por un método.
    * shared_rows: Candidatos del pool común de la misma WSI.
    ***
    Traza cada selección al pool común mediante coordenadas, nivel, tamaño e índice
    TIAToolbox. Rechaza selecciones ausentes o duplicadas.

    Retorna los identificadores de candidatos y la huella canónica de la selección.
    """
    shared_by_key = {
        (
            row["x_level0"],
            row["y_level0"],
            row["level"],
            row["patch_size"],
            row["tiatoolbox_index"],
        ): row
        for row in shared_rows
    }
    resolved: list[dict[str, str]] = []
    for row in selected_rows:
        key = (
            row["x_level0"],
            row["y_level0"],
            "0",
            row["patch_size"],
            row["tiatoolbox_index"],
        )
        candidate = shared_by_key.get(key)
        if candidate is None or candidate["case_id"] != case_id:
            raise ValueError(f"El patch seleccionado no pertenece al pool común: {case_id} {key}.")
        resolved.append(candidate)
    candidate_ids = [row["candidate_id"] for row in resolved]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"El caso {case_id} contiene un candidato seleccionado duplicado.")
    fingerprint_rows = sorted(
        (
            row["case_id"],
            row["x_level0"],
            row["y_level0"],
            row["level"],
            row["patch_size"],
            row["candidate_id"],
            row["tiatoolbox_index"],
        )
        for row in resolved
    )
    return candidate_ids, canonical_hash(fingerprint_rows)


def analyze_case(case_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    ***
    * case_dir: Carpeta de un caso con baseline, v4.1, pool común y cache UNI.
    ***
    Valida el cache, recupera los embeddings exactos de los patches elegidos y calcula
    las métricas morfológicas de ambos métodos.

    Retorna las filas por método y el detalle de las validaciones realizadas.
    """
    case_id = case_dir.name
    cache_path = case_dir / "v4_1" / "embedding_cache.npz"
    cache_metadata_path = case_dir / "v4_1" / "embedding_cache_metadata.json"
    embeddings, cached_ids, cache_metadata = load_embedding_cache(
        cache_path=cache_path,
        metadata_path=cache_metadata_path,
    )
    method_config = json.loads((case_dir / "v4_1" / "method_config.json").read_text())
    config = {
        field: cache_metadata.get(field, method_config.get(field))
        for field in CONFIG_FIELDS
    }
    config_hash = canonical_hash(config)

    if cached_ids != [str(value) for value in cache_metadata.get("candidate_ids", [])]:
        raise ValueError(f"Los IDs del cache y su metadata difieren para {case_id}.")
    if embeddings.ndim != 2 or embeddings.shape != (
        int(cache_metadata["num_embeddings"]),
        int(cache_metadata["embedding_dim"]),
    ):
        raise ValueError(f"Dimensiones inválidas del cache de embeddings para {case_id}: {embeddings.shape}.")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"El cache de embeddings contiene NaN o infinitos para {case_id}.")

    shared_rows = read_csv(case_dir / "shared_candidates.csv")
    cache_index = {candidate_id: index for index, candidate_id in enumerate(cached_ids)}
    rows: list[dict[str, object]] = []
    validation: dict[str, object] = {
        "case_id": case_id,
        "cache_path": str(cache_path),
        "cache_shape": list(embeddings.shape),
        "embedding_configuration_hash": config_hash,
        "methods": {},
    }
    for method_dir, method_name in METHODS.items():
        selected_rows = read_csv(case_dir / method_dir / "selected_metadata.csv")
        if any(row.get("selector") != method_name for row in selected_rows):
            raise ValueError(f"Etiqueta de selector inesperada para {case_id}/{method_name}.")
        candidate_ids, selection_hash = selected_candidate_ids(case_id, selected_rows, shared_rows)
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in cache_index]
        if missing:
            raise ValueError(
                f"Faltan embeddings persistidos para {case_id}/{method_name}: {missing}."
            )
        selected_embeddings = np.asarray(
            [embeddings[cache_index[candidate_id]] for candidate_id in candidate_ids],
            dtype=np.float32,
        )
        metrics, matrix_validation = distance_metrics(selected_embeddings)
        rows.append({
            "case_id": case_id,
            "method": method_name,
            "num_patches_selected": len(selected_rows),
            "num_valid_embeddings": len(selected_embeddings),
            "embedding_dim": selected_embeddings.shape[1],
            **metrics,
            "embedding_references_reused": len(selected_embeddings),
            "embeddings_recomputed": 0,
            "embedding_configuration_hash": config_hash,
            "selected_patch_fingerprint": selection_hash,
            "status": "ok",
        })
        validation["methods"][method_name] = {
            "selected_patch_count": len(selected_rows),
            "valid_embedding_count": len(selected_embeddings),
            "selected_patch_fingerprint": selection_hash,
            **matrix_validation,
        }
    return rows, {"configuration": config, **validation}


def aggregate_rows(per_wsi_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    ***
    * per_wsi_rows: Métricas morfológicas por caso y método.
    ***
    Agrega cada métrica por método y calcula diferencias pareadas v4.1 menos baseline.

    Retorna las filas del CSV agregado y el resumen estructurado para JSON y Markdown.
    """
    by_case = {
        case_id: {str(row["method"]): row for row in per_wsi_rows if row["case_id"] == case_id}
        for case_id in sorted({str(row["case_id"]) for row in per_wsi_rows})
    }
    aggregate_csv: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for metric in METRICS:
        method_stats: dict[str, dict[str, float | int]] = {}
        for method_name in METHODS.values():
            values = [float(methods[method_name][metric]) for methods in by_case.values()]
            stats = descriptive(values)
            method_stats[method_name] = stats
            aggregate_csv.append({
                "row_type": "method_summary",
                "metric": metric,
                "method": method_name,
                **stats,
                "v41_greater_wsi": "",
                "baseline_greater_wsi": "",
                "ties": "",
            })
        differences = [
            float(methods[METHODS["v4_1"]][metric])
            - float(methods[METHODS["baseline"]][metric])
            for methods in by_case.values()
        ]
        v41_greater = sum(value > 1e-12 for value in differences)
        baseline_greater = sum(value < -1e-12 for value in differences)
        ties = len(differences) - v41_greater - baseline_greater
        difference_stats = descriptive(differences)
        aggregate_csv.append({
            "row_type": "paired_difference",
            "metric": metric,
            "method": "v4_1_minus_baseline",
            **difference_stats,
            "v41_greater_wsi": v41_greater,
            "baseline_greater_wsi": baseline_greater,
            "ties": ties,
        })
        summary[metric] = {
            "baseline": method_stats[METHODS["baseline"]],
            "v4_1": method_stats[METHODS["v4_1"]],
            "paired_difference_v4_1_minus_baseline": difference_stats,
            "comparison_counts": {
                "v4_1_greater": v41_greater,
                "baseline_greater": baseline_greater,
                "ties": ties,
            },
        }
    for methods in by_case.values():
        pairwise_delta = (
            float(methods[METHODS["v4_1"]]["mean_pairwise_cosine_distance"])
            - float(methods[METHODS["baseline"]]["mean_pairwise_cosine_distance"])
        )
        nn_delta = (
            float(methods[METHODS["v4_1"]]["mean_nearest_neighbor_cosine_distance"])
            - float(methods[METHODS["baseline"]]["mean_nearest_neighbor_cosine_distance"])
        )
        for row in methods.values():
            row["delta_mean_pairwise_v41_minus_baseline"] = pairwise_delta
            row["delta_mean_nearest_neighbor_v41_minus_baseline"] = nn_delta
    return aggregate_csv, summary


def format_stats(stats: dict[str, float | int]) -> tuple[str, str]:
    """Formatea media con desviación y mediana con cuartiles sin alterar los valores."""
    return (
        f"{float(stats['mean']):.6f} ± {float(stats['sd']):.6f}",
        f"{float(stats['median']):.6f} [{float(stats['q1']):.6f}, {float(stats['q3']):.6f}]",
    )


def render_markdown(summary: dict[str, Any], case_differences: dict[str, dict[str, float]]) -> str:
    """
    ***
    * summary: Estadísticos agregados de diversidad morfológica.
    * case_differences: Diferencias pareadas por WSI.
    ***
    Genera la tabla Markdown versionada con resultados agregados y por caso.

    Retorna el documento completo como texto.
    """
    lines = [
        "# Diversidad morfológica aproximada en el espacio UNI",
        "",
        "Se analizaron los 16 patches previamente seleccionados por método en cada una de las nueve WSI. v4.1 corresponde al método propuesto por nosotros. ",
        "",
        "## Comparación agregada",
        "",
        "| Métrica | Baseline: media ± DE | v4.1: media ± DE | Baseline: mediana [Q1, Q3] | v4.1: mediana [Q1, Q3] | Δ pareada media | Δ pareada mediana | v4.1 / baseline / empates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, label in METRICS.items():
        result = summary[metric]
        baseline_mean, baseline_median = format_stats(result["baseline"])
        v41_mean, v41_median = format_stats(result["v4_1"])
        difference = result["paired_difference_v4_1_minus_baseline"]
        counts = result["comparison_counts"]
        lines.append(
            f"| {label} | {baseline_mean} | {v41_mean} | {baseline_median} | {v41_median} | "
            f"{float(difference['mean']):+.6f} | {float(difference['median']):+.6f} | "
            f"{counts['v4_1_greater']} / {counts['baseline_greater']} / {counts['ties']} |"
        )
    lines.extend([
        "",
        "Δ corresponde a v4.1 menos baseline. DE: desviación estándar; Q1 y Q3: cuartiles 25 y 75.",
        "",
        "## Diferencias pareadas por WSI",
        "",
        "| WSI | Δ distancia media entre pares | Δ distancia media al vecino más cercano |",
        "|---|---:|---:|",
    ])
    for case_id, differences in sorted(case_differences.items()):
        lines.append(
            f"| {case_id} | {differences['mean_pairwise_cosine_distance']:+.6f} | "
            f"{differences['mean_nearest_neighbor_cosine_distance']:+.6f} |"
        )

    pairwise = summary["mean_pairwise_cosine_distance"]
    nearest = summary["mean_nearest_neighbor_cosine_distance"]
    lines.extend([
        "",
        "## Interpretación de los resultados",
        "",
        f"En la distancia coseno media entre pares, v4.1 obtuvo un valor mayor en "
        f"{pairwise['comparison_counts']['v4_1_greater']} de 9 WSI, mientras que el baseline fue mayor en "
        f"{pairwise['comparison_counts']['baseline_greater']} y se observaron {pairwise['comparison_counts']['ties']} empates. "
        f"La diferencia pareada media fue {pairwise['paired_difference_v4_1_minus_baseline']['mean']:+.6f}.",
        "",
        f"Para la distancia media al vecino morfológico más cercano, v4.1 fue mayor en "
        f"{nearest['comparison_counts']['v4_1_greater']} de 9 WSI, el baseline fue mayor en "
        f"{nearest['comparison_counts']['baseline_greater']} y se observaron {nearest['comparison_counts']['ties']} empates. "
        f"La diferencia pareada media fue {nearest['paired_difference_v4_1_minus_baseline']['mean']:+.6f}.",
    ])
    return "\n".join(lines)


def self_check() -> None:
    """Verifica distancias coseno conocidas sobre tres embeddings sintéticos."""
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    metrics, validation = distance_metrics(embeddings)
    assert math.isclose(metrics["mean_pairwise_cosine_distance"], 4.0 / 3.0, abs_tol=1e-7)
    assert math.isclose(metrics["mean_nearest_neighbor_cosine_distance"], 1.0, abs_tol=1e-7)
    assert all(validation.values())
    print("Autocomprobación de diversidad morfológica superada.")


def main() -> int:
    """
    Valida los nueve casos, reutiliza los caches UNI y escribe métricas morfológicas
    por WSI, agregados, resumen JSON y tabla Markdown.

    Retorna cero cuando todas las validaciones y escrituras finalizan correctamente.
    """
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    experiment_dir = args.experiment_dir.expanduser().resolve()
    case_dirs = sorted(
        path for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "baseline/selected_metadata.csv").exists()
        and (path / "v4_1/selected_metadata.csv").exists()
    )
    if len(case_dirs) != args.expected_cases:
        raise RuntimeError(f"Se esperaban {args.expected_cases} casos y se encontraron {len(case_dirs)}.")

    per_wsi_rows: list[dict[str, object]] = []
    case_validations: list[dict[str, object]] = []
    for case_dir in case_dirs:
        rows, validation = analyze_case(case_dir)
        per_wsi_rows.extend(rows)
        case_validations.append(validation)

    config_hashes = {str(row["embedding_configuration_hash"]) for row in per_wsi_rows}
    if len(config_hashes) != 1:
        raise RuntimeError("Los embeddings seleccionados fueron generados con configuraciones UNI distintas.")
    aggregate_csv, aggregate_summary = aggregate_rows(per_wsi_rows)
    case_differences = {
        case_dir.name: {
            "mean_pairwise_cosine_distance": float(next(
                row["delta_mean_pairwise_v41_minus_baseline"]
                for row in per_wsi_rows if row["case_id"] == case_dir.name
            )),
            "mean_nearest_neighbor_cosine_distance": float(next(
                row["delta_mean_nearest_neighbor_v41_minus_baseline"]
                for row in per_wsi_rows if row["case_id"] == case_dir.name
            )),
        }
        for case_dir in case_dirs
    }
    output_dir = experiment_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_manifest(
        per_wsi_rows,
        output_dir / "morphological_diversity_per_wsi.csv",
        PER_WSI_FIELDS,
    )
    write_csv_manifest(
        aggregate_csv,
        output_dir / "morphological_diversity_aggregate.csv",
        AGGREGATE_FIELDS,
    )
    summary = {
        "status": "completed",
        "metric": {
            "distance": "cosine_distance_on_l2_normalized_embeddings",
            "pairwise": "upper_triangle_without_diagonal_each_pair_once",
            "nearest_neighbor": "row_minimum_excluding_diagonal",
            "standard_deviation": "sample_sd_ddof_1",
            "quartiles": "numpy_linear_percentiles_25_75",
        },
        "cohort": {
            "cases_expected": args.expected_cases,
            "cases_processed": len(case_dirs),
            "case_ids": [case_dir.name for case_dir in case_dirs],
            "methods": list(METHODS.values()),
        },
        "embedding_configuration": case_validations[0]["configuration"],
        "embedding_configuration_hash": next(iter(config_hashes)),
        "validation": {
            "same_uni_configuration_for_all_cases_and_methods": True,
            "original_selected_patches_traced_exactly": True,
            "all_distance_matrices_symmetric": True,
            "all_distance_matrix_diagonals_zero_within_tolerance": True,
            "all_distances_finite_and_within_cosine_bounds": True,
            "distance_tolerance": TOLERANCE,
            "selected_embedding_references_reused": sum(
                int(row["embedding_references_reused"]) for row in per_wsi_rows
            ),
            "selected_embeddings_recomputed": 0,
            "case_details": case_validations,
            "errors": [],
        },
        "per_wsi_differences_v4_1_minus_baseline": case_differences,
        "aggregate": aggregate_summary,
    }
    write_json_manifest(summary, output_dir / "morphological_diversity_summary.json")
    (output_dir / "morphological_diversity_results.md").write_text(
        render_markdown(aggregate_summary, case_differences),
        encoding="utf-8",
    )
    print(f"Casos procesados: {len(case_dirs)}")
    print(f"Referencias de embeddings reutilizadas: {summary['validation']['selected_embedding_references_reused']}")
    print("Embeddings recalculados: 0")
    print(f"Resultados: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
