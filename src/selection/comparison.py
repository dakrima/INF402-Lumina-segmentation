"""Comparación técnica de selecciones sin ejecutar modelos adicionales."""

from __future__ import annotations

import csv
import importlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.selection.quality_filters import compute_patch_features


REQUIRED_RESULT_FILES = [
    "candidate_metadata.csv",
    "selected_metadata.csv",
    "selection_summary.json",
    "method_config.json",
    "patch_selection_preview.png",
]
REQUIRED_RESULT_DIRS = ["selected"]
COMPARISON_METRICS_FIELDS = [
    "metric",
    "baseline_value",
    "smart_value",
    "delta_smart_minus_baseline",
    "relative_delta",
    "higher_is_better",
    "interpretation",
]
SELECTED_OVERLAP_FIELDS = [
    "candidate_key",
    "x_level0",
    "y_level0",
    "in_baseline",
    "in_smart",
    "baseline_rank",
    "smart_rank",
]
COMPARISON_SELECTED_PATCH_FIELDS = [
    "method",
    "patch_id",
    "filename",
    "candidate_key",
    "rank",
    "x_level0",
    "y_level0",
    "tissue_ratio_metadata",
    "tissue_ratio_recomputed",
    "nuclear_signal_recomputed",
    "nuclear_signal_rgb_recomputed",
    "nuclear_signal_hed_recomputed",
    "visual_entropy_recomputed",
    "blur_score_recomputed",
    "artifact_penalty_recomputed",
    "score_raw",
    "score_final",
]
MEDICAL_COMPARISON_FIELDS = [
    "medical_image_quality_score",
    "medical_image_utility_score",
    "medical_texture_score",
    "medical_pseudo_cellularity_score",
    "medical_artifact_penalty",
]
SHARED_CONFIG_FIELDS = [
    "wsi_path",
    "patch_size",
    "stride",
    "max_patches",
    "seed",
    "candidate_pool",
    "candidate_metadata_semantics",
    "num_candidate_rows_written",
    "candidate_pool_count",
    "candidate_pool_hash",
]


@dataclass(frozen=True)
class SelectorRun:
    """Manifiestos cargados de una corrida de selección."""

    label: str
    directory: Path
    summary: dict[str, Any]
    method_config: dict[str, Any]
    candidate_rows: list[dict[str, str]]
    selected_rows: list[dict[str, str]]


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuración para comparar las salidas de ambos métodos."""

    baseline_dir: Path
    smart_dir: Path
    output_dir: Path
    root_dir: Path
    feature_size: int = 256
    overwrite: bool = False
    recompute_selected_features: bool = True
    require_exact_candidate_pool: bool = False


def _utc_now_iso() -> str:
    """Retorna un timestamp UTC ISO-8601 para el resumen de comparación."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_path(path: Path, root_dir: Path) -> Path:
    """Resuelve rutas relativas respecto de la raíz del repositorio."""
    if path.is_absolute():
        return path.expanduser().resolve()
    return (root_dir / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Indica si una ruta se encuentra dentro de otra."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_output_dir(output_dir: Path, root_dir: Path, overwrite: bool) -> None:
    """Valida y prepara una carpeta de comparación sin borrar rutas críticas."""
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    if resolved_output.exists() and any(child.name != ".gitkeep" for child in resolved_output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"La carpeta de salida ya existe y no está vacía: {resolved_output}. "
                "Use --overwrite para regenerar la comparación."
            )
        dangerous_paths = {
            Path("/").resolve(),
            Path.home().resolve(),
            resolved_root,
            resolved_root / "data",
            resolved_root / "outputs",
        }
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite solo limpia carpetas de salida dentro del repositorio.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Se rechazó una ruta de salida peligrosa: {resolved_output}")
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    """Lee un archivo JSON y retorna su objeto raíz."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Lee un CSV completo y retorna sus filas como diccionarios."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    """Guarda JSON indentado y retorna la ruta generada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> Path:
    """Guarda filas con columnas estables y retorna la ruta del CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validate_result_dir(result_dir: Path) -> None:
    """Comprueba que una corrida contenga manifests, configuración y patches."""
    if not result_dir.exists():
        raise FileNotFoundError(f"La carpeta de salida del selector no existe: {result_dir}")
    for file_name in REQUIRED_RESULT_FILES:
        path = result_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Falta el archivo requerido: {path}")
    for dir_name in REQUIRED_RESULT_DIRS:
        path = result_dir / dir_name
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Falta la carpeta requerida: {path}")


def load_selector_run(label: str, directory: Path) -> SelectorRun:
    """
    ***
    * label: Etiqueta interna usada en la comparación.
    * directory: Carpeta generada por un selector.
    ***
    Valida y carga summary, configuración, candidatos y patches seleccionados.

    Retorna una representación inmutable de la corrida.
    """
    _validate_result_dir(directory)
    return SelectorRun(
        label=label,
        directory=directory,
        summary=_read_json(directory / "selection_summary.json"),
        method_config=_read_json(directory / "method_config.json"),
        candidate_rows=_read_csv(directory / "candidate_metadata.csv"),
        selected_rows=_read_csv(directory / "selected_metadata.csv"),
    )


def _to_float(value: object) -> float | None:
    """Convierte un valor finito a `float` o retorna `None`."""
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_int(value: object) -> int | None:
    """Convierte texto o número a entero o retorna `None`."""
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _candidate_key_from_xy(x_level0: object, y_level0: object) -> str:
    """Construye la clave estable `x_y` usada para comparar coordenadas."""
    return f"x{int(float(x_level0))}_y{int(float(y_level0))}"


def _selector_title(run: SelectorRun) -> str:
    """Obtiene el nombre persistido del selector de una corrida."""
    selector = run.summary.get("selector") or run.method_config.get("selector")
    return str(selector or run.label)


def _candidate_lookup_by_xy(candidate_rows: list[dict[str, str]]) -> dict[tuple[int, int], str]:
    """Indexa IDs de candidatos por sus coordenadas de nivel 0."""
    lookup: dict[tuple[int, int], str] = {}
    for row in candidate_rows:
        x_level0 = _to_int(row.get("x_level0"))
        y_level0 = _to_int(row.get("y_level0"))
        candidate_id = row.get("candidate_id")
        if x_level0 is None or y_level0 is None or not candidate_id:
            continue
        lookup[(x_level0, y_level0)] = candidate_id
    return lookup


def _selected_records(run: SelectorRun) -> list[dict[str, object]]:
    """Normaliza las filas seleccionadas para cálculos comparativos."""
    candidate_by_xy = _candidate_lookup_by_xy(run.candidate_rows)
    records: list[dict[str, object]] = []
    for row in run.selected_rows:
        x_level0 = _to_int(row.get("x_level0"))
        y_level0 = _to_int(row.get("y_level0"))
        if x_level0 is None or y_level0 is None:
            continue
        candidate_key = candidate_by_xy.get(
            (x_level0, y_level0),
            row.get("candidate_id") or _candidate_key_from_xy(x_level0, y_level0),
        )
        records.append(
            {
                "method": run.label,
                "candidate_key": candidate_key,
                "patch_id": row.get("patch_id") or Path(row.get("filename", "")).stem,
                "filename": row.get("filename", ""),
                "rank": row.get("rank", ""),
                "x_level0": x_level0,
                "y_level0": y_level0,
                "patch_size": _to_int(row.get("patch_size")) or _to_int(run.summary.get("patch_size")) or 0,
                "tissue_ratio_metadata": row.get("tissue_ratio", ""),
                "score_raw": row.get("score_raw", ""),
                "score_final": row.get("score_final", ""),
                "source_row": row,
            }
        )
    return records


def _candidate_pool_keys(run: SelectorRun) -> set[str]:
    """Retorna las claves canónicas del pool de candidatos."""
    keys: set[str] = set()
    for row in run.candidate_rows:
        candidate_id = row.get("candidate_id")
        if candidate_id:
            keys.add(candidate_id)
            continue
        x_level0 = row.get("x_level0")
        y_level0 = row.get("y_level0")
        if x_level0 not in ("", None) and y_level0 not in ("", None):
            keys.add(_candidate_key_from_xy(x_level0, y_level0))
    return keys


def _candidate_pool_coordinates(run: SelectorRun) -> list[tuple[int, int, int, int]]:
    """Retorna la geometría ordenada del pool para validar igualdad exacta."""
    coordinates: list[tuple[int, int, int, int]] = []
    for row in run.candidate_rows:
        x_level0 = _to_int(row.get("x_level0"))
        y_level0 = _to_int(row.get("y_level0"))
        patch_size = _to_int(row.get("patch_size")) or _to_int(run.summary.get("patch_size"))
        if x_level0 is None or y_level0 is None or patch_size is None:
            continue
        coordinates.append((x_level0, y_level0, 0, patch_size))
    return sorted(coordinates)


def validate_shared_config(baseline: SelectorRun, smart: SelectorRun) -> tuple[dict[str, Any], list[str]]:
    """
    Compara presupuesto, tamaño, stride, semilla, máscara y pool de candidatos.

    Retorna las comprobaciones de igualdad y advertencias técnicas ante diferencias.
    """
    warnings: list[str] = []
    shared_config: dict[str, Any] = {}
    for field_name in SHARED_CONFIG_FIELDS:
        baseline_value = baseline.summary.get(field_name)
        smart_value = smart.summary.get(field_name)
        shared_config[field_name] = {
            "baseline": baseline_value,
            "smart": smart_value,
            "matches": baseline_value == smart_value,
        }
        if baseline_value != smart_value:
            warnings.append(
                f"El campo compartido {field_name} difiere: "
                f"baseline={baseline_value!r}, v4.1={smart_value!r}."
            )

    baseline_pool_keys = _candidate_pool_keys(baseline)
    smart_pool_keys = _candidate_pool_keys(smart)
    pool_matches = baseline_pool_keys == smart_pool_keys
    shared_config["candidate_pool_keys"] = {
        "baseline_count": len(baseline_pool_keys),
        "smart_count": len(smart_pool_keys),
        "matches": pool_matches,
    }
    if not pool_matches:
        warnings.append(
            "Las claves del pool difieren entre métodos; la comparación podría usar pools distintos."
        )
    baseline_hash = baseline.summary.get("candidate_pool_hash")
    smart_hash = smart.summary.get("candidate_pool_hash")
    hash_matches = bool(baseline_hash) and baseline_hash == smart_hash
    shared_config["candidate_pool_hash_exact"] = {
        "baseline": baseline_hash,
        "smart": smart_hash,
        "matches": hash_matches,
    }
    if not hash_matches:
        warnings.append("Los hashes SHA-256 del pool faltan o difieren entre métodos.")

    baseline_coordinates = _candidate_pool_coordinates(baseline)
    smart_coordinates = _candidate_pool_coordinates(smart)
    coordinates_match = baseline_coordinates == smart_coordinates
    shared_config["candidate_pool_coordinates_exact"] = {
        "baseline_count": len(baseline_coordinates),
        "smart_count": len(smart_coordinates),
        "matches": coordinates_match,
    }
    if not coordinates_match:
        warnings.append("Las coordenadas del pool difieren entre métodos.")
    return shared_config, warnings


def compute_overlap_metrics(
    baseline_records: list[dict[str, object]],
    smart_records: list[dict[str, object]],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Calcula intersección de coordenadas y retorna métricas y filas de solapamiento."""
    baseline_by_key = {str(row["candidate_key"]): row for row in baseline_records}
    smart_by_key = {str(row["candidate_key"]): row for row in smart_records}
    baseline_keys = set(baseline_by_key)
    smart_keys = set(smart_by_key)
    overlap_keys = baseline_keys & smart_keys
    union_keys = baseline_keys | smart_keys

    overlap_rows: list[dict[str, object]] = []
    for key in sorted(union_keys):
        baseline_row = baseline_by_key.get(key)
        smart_row = smart_by_key.get(key)
        source = baseline_row or smart_row or {}
        overlap_rows.append(
            {
                "candidate_key": key,
                "x_level0": source.get("x_level0", ""),
                "y_level0": source.get("y_level0", ""),
                "in_baseline": key in baseline_keys,
                "in_smart": key in smart_keys,
                "baseline_rank": baseline_row.get("rank", "") if baseline_row else "",
                "smart_rank": smart_row.get("rank", "") if smart_row else "",
            }
        )

    overlap_metrics = {
        "num_selected_baseline": len(baseline_keys),
        "num_selected_smart": len(smart_keys),
        "num_overlap_selected": len(overlap_keys),
        "overlap_ratio_baseline": len(overlap_keys) / len(baseline_keys) if baseline_keys else None,
        "overlap_ratio_smart": len(overlap_keys) / len(smart_keys) if smart_keys else None,
        "jaccard_selected": len(overlap_keys) / len(union_keys) if union_keys else None,
    }
    return overlap_metrics, overlap_rows


def _stats(values: list[float]) -> dict[str, float | None]:
    """Resume media, mediana, extremos y desviación poblacional de una lista."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _method_feature_stats(
    rows: list[dict[str, object]],
    feature_names: list[str],
) -> dict[str, dict[str, float | None]]:
    """Agrupa estadísticos por feature recomputada para un método."""
    result: dict[str, dict[str, float | None]] = {}
    for feature_name in feature_names:
        values = [
            value for value in (_to_float(row.get(feature_name)) for row in rows)
            if value is not None
        ]
        result[feature_name] = _stats(values)
    return result


def _selected_metadata_stats(run: SelectorRun, field_names: list[str]) -> dict[str, dict[str, float | None]]:
    """Resume campos numéricos ya persistidos en la selección."""
    return _method_feature_stats(run.selected_rows, field_names)


def _selected_cluster_metrics(run: SelectorRun) -> dict[str, float | None]:
    """Calcula cobertura y entropía normalizada de clusters seleccionados."""
    cluster_ids = [
        str(row.get("embedding_cluster_id", ""))
        for row in run.selected_rows
        if str(row.get("embedding_cluster_id", "")).strip()
    ]
    if not cluster_ids:
        return {
            "num_embedding_clusters_covered": None,
            "selected_cluster_entropy": None,
        }
    counts: dict[str, int] = {}
    for cluster_id in cluster_ids:
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    total = float(sum(counts.values()))
    probabilities = [count / total for count in counts.values() if count > 0]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return {
        "num_embedding_clusters_covered": float(len(counts)),
        "selected_cluster_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
    }


def _embedding_distance_metrics(run: SelectorRun) -> dict[str, float | None]:
    """Calcula distancias UNI si la corrida conserva un cache compatible."""
    empty = {
        "mean_pairwise_embedding_distance": None,
        "median_pairwise_embedding_distance": None,
        "min_pairwise_embedding_distance": None,
    }
    cache_path_value = run.summary.get("embedding_cache_path") or run.method_config.get("embedding_cache_path")
    if not cache_path_value:
        return empty
    cache_path = Path(str(cache_path_value)).expanduser()
    if not cache_path.exists():
        return empty
    try:
        import numpy as np  # type: ignore
    except Exception:
        return empty

    try:
        with np.load(cache_path, allow_pickle=True) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            candidate_ids = [str(value) for value in data["candidate_ids"].tolist()]
    except Exception:
        return empty
    if embeddings.ndim != 2 or embeddings.shape[0] != len(candidate_ids):
        return empty
    candidate_by_xy = _candidate_lookup_by_xy(run.candidate_rows)
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    selected_indices: list[int] = []
    for row in run.selected_rows:
        x_level0 = _to_int(row.get("x_level0"))
        y_level0 = _to_int(row.get("y_level0"))
        if x_level0 is None or y_level0 is None:
            continue
        candidate_id = candidate_by_xy.get((x_level0, y_level0))
        if candidate_id in index_by_id:
            selected_indices.append(index_by_id[candidate_id])
    if len(selected_indices) < 2:
        return empty
    selected_embeddings = embeddings[selected_indices]
    norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
    selected_embeddings = selected_embeddings / np.where(norms <= 1e-12, 1.0, norms)
    similarities = selected_embeddings @ selected_embeddings.T
    distances: list[float] = []
    for row_index in range(similarities.shape[0]):
        for col_index in range(row_index + 1, similarities.shape[1]):
            distances.append(float(max(0.0, min(2.0, 1.0 - similarities[row_index, col_index]))))
    if not distances:
        return empty
    return {
        "mean_pairwise_embedding_distance": float(statistics.mean(distances)),
        "median_pairwise_embedding_distance": float(statistics.median(distances)),
        "min_pairwise_embedding_distance": float(min(distances)),
    }


def compute_optional_selector_metrics(baseline: SelectorRun, smart: SelectorRun) -> dict[str, Any]:
    """Calcula métricas opcionales cuando existen metadata y cache compatibles."""
    return {
        "baseline": {
            "medical": _selected_metadata_stats(baseline, MEDICAL_COMPARISON_FIELDS),
            "embedding_clusters": _selected_cluster_metrics(baseline),
            "embedding_distances": _embedding_distance_metrics(baseline),
        },
        "smart": {
            "medical": _selected_metadata_stats(smart, MEDICAL_COMPARISON_FIELDS),
            "embedding_clusters": _selected_cluster_metrics(smart),
            "embedding_distances": _embedding_distance_metrics(smart),
        },
    }


def recompute_selected_patch_features(
    run: SelectorRun,
    records: list[dict[str, object]],
    *,
    feature_size: int,
    recompute: bool,
) -> list[dict[str, object]]:
    """
    Recalcula las features comparables leyendo un PNG seleccionado a la vez.

    Conserva los campos persistidos y retorna una fila por patch y método.
    """
    rows: list[dict[str, object]] = []
    selected_dir = run.directory / "selected"
    for record in records:
        output_row: dict[str, object] = {
            "method": run.label,
            "patch_id": record["patch_id"],
            "filename": record["filename"],
            "candidate_key": record["candidate_key"],
            "rank": record["rank"],
            "x_level0": record["x_level0"],
            "y_level0": record["y_level0"],
            "tissue_ratio_metadata": record["tissue_ratio_metadata"],
            "tissue_ratio_recomputed": "",
            "nuclear_signal_recomputed": "",
            "nuclear_signal_rgb_recomputed": "",
            "nuclear_signal_hed_recomputed": "",
            "visual_entropy_recomputed": "",
            "blur_score_recomputed": "",
            "artifact_penalty_recomputed": "",
            "score_raw": record["score_raw"],
            "score_final": record["score_final"],
        }
        if recompute:
            filename = str(record["filename"])
            patch_path = selected_dir / filename
            if not patch_path.exists():
                raise FileNotFoundError(f"Falta el patch PNG seleccionado: {patch_path}")
            with Image.open(patch_path) as image:
                features_rgb = compute_patch_features(
                    rgb_patch=image.convert("RGB"),
                    feature_size=feature_size,
                    nuclear_proxy="rgb_purple",
                )
                features_hed = compute_patch_features(
                    rgb_patch=image.convert("RGB"),
                    feature_size=feature_size,
                    nuclear_proxy="hed_deconvolution",
                )
            output_row.update(
                {
                    "tissue_ratio_recomputed": f"{features_rgb['tissue_ratio']:.6f}",
                    "nuclear_signal_recomputed": f"{features_rgb['nuclear_signal']:.6f}",
                    "nuclear_signal_rgb_recomputed": f"{features_rgb['nuclear_signal']:.6f}",
                    "nuclear_signal_hed_recomputed": f"{features_hed['nuclear_signal']:.6f}",
                    "visual_entropy_recomputed": f"{features_rgb['visual_entropy']:.6f}",
                    "blur_score_recomputed": f"{features_rgb['blur_score']:.6f}",
                    "artifact_penalty_recomputed": f"{features_rgb['artifact_penalty']:.6f}",
                }
            )
        rows.append(output_row)
    return rows


def _pairwise_distances(records: list[dict[str, object]]) -> list[float]:
    """Calcula una vez cada distancia euclidiana entre centros de patches."""
    distances: list[float] = []
    centers = []
    for row in records:
        patch_size = float(row.get("patch_size") or 0)
        centers.append(
            (
                float(row["x_level0"]) + patch_size / 2.0,
                float(row["y_level0"]) + patch_size / 2.0,
            )
        )
    for index, (x0, y0) in enumerate(centers):
        for x1, y1 in centers[index + 1:]:
            distances.append(float(math.hypot(x0 - x1, y0 - y1)))
    return distances


def compute_spatial_metrics(
    records: list[dict[str, object]],
    summary: dict[str, Any],
) -> dict[str, float | None]:
    """Calcula distancias y cobertura espacial aproximada de una selección."""
    distances = _pairwise_distances(records)
    nearest_distances: list[float] = []
    for row in records:
        patch_size = float(row.get("patch_size") or 0)
        cx = float(row["x_level0"]) + patch_size / 2.0
        cy = float(row["y_level0"]) + patch_size / 2.0
        other_distances = []
        for other in records:
            if other is row:
                continue
            other_patch_size = float(other.get("patch_size") or 0)
            ox = float(other["x_level0"]) + other_patch_size / 2.0
            oy = float(other["y_level0"]) + other_patch_size / 2.0
            other_distances.append(float(math.hypot(cx - ox, cy - oy)))
        if other_distances:
            nearest_distances.append(min(other_distances))

    centers_x = [
        float(row["x_level0"]) + float(row.get("patch_size") or 0) / 2.0
        for row in records
    ]
    centers_y = [
        float(row["y_level0"]) + float(row.get("patch_size") or 0) / 2.0
        for row in records
    ]
    bbox_area = (
        (max(centers_x) - min(centers_x)) * (max(centers_y) - min(centers_y))
        if centers_x and centers_y
        else None
    )
    slide_width = _to_float(summary.get("slide_width"))
    slide_height = _to_float(summary.get("slide_height"))
    slide_area = slide_width * slide_height if slide_width and slide_height else None
    coverage_ratio = bbox_area / slide_area if bbox_area is not None and slide_area else None

    return {
        "mean_pairwise_distance": float(statistics.mean(distances)) if distances else None,
        "median_pairwise_distance": float(statistics.median(distances)) if distances else None,
        "min_pairwise_distance": float(min(distances)) if distances else None,
        "mean_nearest_neighbor_distance": float(statistics.mean(nearest_distances))
        if nearest_distances
        else None,
        "median_nearest_neighbor_distance": float(statistics.median(nearest_distances))
        if nearest_distances
        else None,
        "min_nearest_neighbor_distance": float(min(nearest_distances))
        if nearest_distances
        else None,
        "spatial_bbox_area": float(bbox_area) if bbox_area is not None else None,
        "spatial_coverage_ratio_approx": float(coverage_ratio)
        if coverage_ratio is not None
        else None,
    }


def _metric_row(
    metric: str,
    baseline_value: object,
    smart_value: object,
    *,
    higher_is_better: bool | None,
    interpretation: str,
) -> dict[str, object]:
    """Construye una fila comparable con valores, diferencia y dirección esperada."""
    baseline_float = _to_float(baseline_value)
    smart_float = _to_float(smart_value)
    delta = (
        smart_float - baseline_float
        if baseline_float is not None and smart_float is not None
        else None
    )
    relative_delta = (
        delta / baseline_float
        if delta is not None and baseline_float not in (None, 0)
        else None
    )
    return {
        "metric": metric,
        "baseline_value": baseline_value,
        "smart_value": smart_value,
        "delta_smart_minus_baseline": delta if delta is not None else "",
        "relative_delta": relative_delta if relative_delta is not None else "",
        "higher_is_better": higher_is_better if higher_is_better is not None else "",
        "interpretation": interpretation,
    }


def build_comparison_metrics_rows(
    baseline: SelectorRun,
    smart: SelectorRun,
    overlap_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
    spatial_metrics: dict[str, Any],
    optional_selector_metrics: dict[str, Any],
) -> list[dict[str, object]]:
    """Combina métricas técnicas, espaciales y opcionales en filas estables."""
    rows: list[dict[str, object]] = []
    count_metrics = [
        "num_candidates_generated",
        "num_thumbnail_candidates_passing_mask",
        "num_candidate_rows_written",
        "num_candidates_evaluated",
        "num_candidates_scored",
        "num_selected",
        "regions_covered",
        "active_regions",
    ]
    for metric in count_metrics:
        rows.append(
            _metric_row(
                metric,
                baseline.summary.get(metric, 0),
                smart.summary.get(metric, 0),
                higher_is_better=None,
                interpretation="Conteo operativo; las diferencias dependen del diseño de cada selector.",
            )
        )
    rows.append(
        _metric_row(
            "runtime_seconds",
            baseline.summary.get("runtime_seconds"),
            smart.summary.get("runtime_seconds"),
            higher_is_better=False,
            interpretation="Un menor tiempo de ejecución reduce el costo operativo.",
        )
    )
    for metric in [
        "quota_fill_rate",
        "mean_feature_diversity_bonus_selected",
    ]:
        rows.append(
            _metric_row(
                metric,
                baseline.summary.get(metric),
                smart.summary.get(metric),
                higher_is_better=True,
                interpretation=(
                    "Métrica de diversidad o cuotas propia del selector; comparar solo cuando exista."
                ),
            )
        )
    for metric in [
        "nuclear_proxy",
        "spatial_strategy",
        "diversity_strategy",
    ]:
        rows.append(
            _metric_row(
                metric,
                baseline.summary.get(metric, ""),
                smart.summary.get(metric, ""),
                higher_is_better=None,
                interpretation="Campo descriptivo de configuración del selector.",
            )
        )
    for metric in [
        "num_overlap_selected",
        "overlap_ratio_baseline",
        "overlap_ratio_smart",
        "jaccard_selected",
    ]:
        value = overlap_metrics.get(metric)
        rows.append(
            _metric_row(
                metric,
                value,
                value,
                higher_is_better=None,
                interpretation="El solapamiento es descriptivo; un valor mayor no implica mejora.",
            )
        )

    feature_specs = [
        ("tissue_ratio_recomputed", True, "Una mayor fracción de tejido puede indicar menos fondo vacío."),
        (
            "nuclear_signal_recomputed",
            True,
            "Proxy nuclear RGB heredado y conservado por compatibilidad.",
        ),
        (
            "nuclear_signal_rgb_recomputed",
            True,
            "Una señal RGB mayor puede indicar más regiones azul-morado asociadas a hematoxilina.",
        ),
        (
            "nuclear_signal_hed_recomputed",
            True,
            "Una señal HED mayor puede indicar más señal de hematoxilina basada en tinción.",
        ),
        ("visual_entropy_recomputed", True, "Una entropía mayor puede indicar más variación visual."),
        ("blur_score_recomputed", True, "Una varianza de gradiente mayor puede indicar más nitidez."),
        ("artifact_penalty_recomputed", False, "Una menor penalización por artefactos es preferible."),
    ]
    for feature_name, higher_is_better, interpretation in feature_specs:
        rows.append(
            _metric_row(
                f"mean_{feature_name}",
                feature_metrics["baseline"][feature_name]["mean"],
                feature_metrics["smart"][feature_name]["mean"],
                higher_is_better=higher_is_better,
                interpretation=interpretation,
            )
        )
    for feature_name in [
        "nuclear_signal_rgb_recomputed",
        "nuclear_signal_hed_recomputed",
    ]:
        rows.append(
            _metric_row(
                f"median_{feature_name}",
                feature_metrics["baseline"][feature_name]["median"],
                feature_metrics["smart"][feature_name]["median"],
                higher_is_better=True,
                interpretation="Mediana del proxy nuclear en los patches seleccionados.",
            )
        )

    for metric in [
        "mean_pairwise_distance",
        "median_pairwise_distance",
        "min_pairwise_distance",
        "mean_nearest_neighbor_distance",
        "median_nearest_neighbor_distance",
        "min_nearest_neighbor_distance",
        "spatial_bbox_area",
        "spatial_coverage_ratio_approx",
    ]:
        rows.append(
            _metric_row(
                metric,
                spatial_metrics["baseline"].get(metric),
                spatial_metrics["smart"].get(metric),
                higher_is_better=True,
                interpretation="Un valor mayor indica una dispersión espacial más amplia.",
            )
        )
    for field_name in MEDICAL_COMPARISON_FIELDS:
        rows.append(
            _metric_row(
                f"mean_{field_name}",
                optional_selector_metrics["baseline"]["medical"][field_name]["mean"],
                optional_selector_metrics["smart"]["medical"][field_name]["mean"],
                higher_is_better=False if field_name == "medical_artifact_penalty" else True,
                interpretation="Proxy técnico opcional de imagen médica utilizado por v4.1.",
            )
        )
    for metric in [
        "num_embedding_clusters_covered",
        "selected_cluster_entropy",
    ]:
        rows.append(
            _metric_row(
                metric,
                optional_selector_metrics["baseline"]["embedding_clusters"].get(metric),
                optional_selector_metrics["smart"]["embedding_clusters"].get(metric),
                higher_is_better=True,
                interpretation="Métrica descriptiva opcional de diversidad entre clusters.",
            )
        )
    for metric in [
        "mean_pairwise_embedding_distance",
        "median_pairwise_embedding_distance",
        "min_pairwise_embedding_distance",
    ]:
        rows.append(
            _metric_row(
                metric,
                optional_selector_metrics["baseline"]["embedding_distances"].get(metric),
                optional_selector_metrics["smart"]["embedding_distances"].get(metric),
                higher_is_better=True,
                interpretation="Métrica opcional de distancia UNI cuando existe un cache compatible.",
            )
        )
    return rows


def _format_float(value: object, digits: int = 3) -> str:
    """Formatea un número opcional para las etiquetas de previews."""
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _feature_mean(
    feature_metrics: dict[str, Any],
    method: str,
    feature_name: str,
) -> float | None:
    """Obtiene la media de una feature para uno de los métodos."""
    return feature_metrics.get(method, {}).get(feature_name, {}).get("mean")


def _preview_nuclear_feature(feature_metrics: dict[str, Any]) -> tuple[str, str]:
    """Selecciona el proxy nuclear disponible y su etiqueta visual."""
    if "nuclear_signal_hed_recomputed" in feature_metrics.get("smart", {}):
        return "nuclear_signal_hed_recomputed", "media nuclear HED"
    return "nuclear_signal_recomputed", "media nuclear"


def _preview_footer_lines(
    overlap_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
) -> list[str]:
    """Construye las líneas de métricas incluidas al pie de la preview."""
    nuclear_feature, nuclear_label = _preview_nuclear_feature(feature_metrics)
    baseline_features = feature_metrics["baseline"]
    smart_features = feature_metrics["smart"]
    return [
        (
            f"seleccionados baseline/v4.1: {overlap_metrics['num_selected_baseline']} / "
            f"{overlap_metrics['num_selected_smart']} | solapamiento: "
            f"{overlap_metrics['num_overlap_selected']} | jaccard: "
            f"{_format_float(overlap_metrics['jaccard_selected'])}"
        ),
        (
            "tejido medio: "
            f"{_format_float(baseline_features['tissue_ratio_recomputed']['mean'])} / "
            f"{_format_float(smart_features['tissue_ratio_recomputed']['mean'])} | "
            f"{nuclear_label}: "
            f"{_format_float(_feature_mean(feature_metrics, 'baseline', nuclear_feature))} / "
            f"{_format_float(_feature_mean(feature_metrics, 'smart', nuclear_feature))} | "
            "artefacto medio: "
            f"{_format_float(baseline_features['artifact_penalty_recomputed']['mean'])} / "
            f"{_format_float(smart_features['artifact_penalty_recomputed']['mean'])}"
        ),
    ]


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    """Redimensiona una imagen conservando su relación de aspecto."""
    ratio = height / image.height
    width = max(1, int(round(image.width * ratio)))
    return image.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)


def _slide_dimensions(run: SelectorRun, records: list[dict[str, object]]) -> tuple[float, float]:
    """Obtiene dimensiones de WSI y usa metadata de filas como fallback."""
    slide_width = _to_float(run.summary.get("slide_width"))
    slide_height = _to_float(run.summary.get("slide_height"))
    if slide_width and slide_height:
        return slide_width, slide_height

    selected_widths = [
        _to_float(row.get("slide_width")) for row in (record.get("source_row", {}) for record in records)
    ]
    selected_heights = [
        _to_float(row.get("slide_height")) for row in (record.get("source_row", {}) for record in records)
    ]
    slide_width = next((value for value in selected_widths if value), None)
    slide_height = next((value for value in selected_heights if value), None)
    if slide_width and slide_height:
        return slide_width, slide_height

    max_x = max(
        (
            float(record["x_level0"]) + float(record.get("patch_size") or 0)
            for record in records
        ),
        default=1.0,
    )
    max_y = max(
        (
            float(record["y_level0"]) + float(record.get("patch_size") or 0)
            for record in records
        ),
        default=1.0,
    )
    return max(max_x, 1.0), max(max_y, 1.0)


def _blank_slide_thumbnail(run: SelectorRun, records: list[dict[str, object]]) -> Image.Image:
    """Crea un lienzo proporcional cuando no es posible abrir la WSI."""
    slide_width, slide_height = _slide_dimensions(run, records)
    thumbnail_max_size = (
        _to_int(run.summary.get("thumbnail_max_size"))
        or _to_int(run.method_config.get("thumbnail_max_size"))
        or 2048
    )
    scale = thumbnail_max_size / max(slide_width, slide_height, 1.0)
    width = max(1, int(round(slide_width * scale)))
    height = max(1, int(round(slide_height * scale)))
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(190, 190, 190), width=2)
    draw.text((12, 12), "thumbnail no disponible", fill=(80, 80, 80), font=ImageFont.load_default())
    return image


def _load_clean_thumbnail(run: SelectorRun, records: list[dict[str, object]]) -> Image.Image:
    """Carga un thumbnail limpio de la WSI o utiliza el lienzo de respaldo."""
    wsi_path = run.summary.get("wsi_path") or run.method_config.get("wsi_path")
    if wsi_path:
        resolved_wsi_path = Path(str(wsi_path)).expanduser()
        if resolved_wsi_path.exists():
            thumbnail_max_size = (
                _to_int(run.summary.get("thumbnail_max_size"))
                or _to_int(run.method_config.get("thumbnail_max_size"))
                or 2048
            )
            try:
                openslide = importlib.import_module("openslide")
                slide = openslide.OpenSlide(str(resolved_wsi_path))
                try:
                    return slide.get_thumbnail((thumbnail_max_size, thumbnail_max_size)).convert("RGB")
                finally:
                    close = getattr(slide, "close", None)
                    if close:
                        close()
            except Exception:
                pass
    return _blank_slide_thumbnail(run, records)


def _draw_selected_boxes_on_thumbnail(
    thumbnail: Image.Image,
    run: SelectorRun,
    records: list[dict[str, object]],
) -> Image.Image:
    """Dibuja las cajas de los patches seleccionados sobre un thumbnail."""
    slide_width, slide_height = _slide_dimensions(run, records)
    scale_x = thumbnail.width / max(slide_width, 1.0)
    scale_y = thumbnail.height / max(slide_height, 1.0)
    draw = ImageDraw.Draw(thumbnail)
    for record in records:
        x0 = float(record["x_level0"]) * scale_x
        y0 = float(record["y_level0"]) * scale_y
        patch_size = float(record.get("patch_size") or run.summary.get("patch_size") or 0)
        x1 = (float(record["x_level0"]) + patch_size) * scale_x
        y1 = (float(record["y_level0"]) + patch_size) * scale_y
        draw.rectangle((x0, y0, x1, y1), outline=(0, 170, 70), width=4)
    return thumbnail


def save_comparison_preview(
    baseline: SelectorRun,
    smart: SelectorRun,
    output_path: Path,
    *,
    overlap_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
) -> Path:
    """Crea una preview lado a lado y retorna la ruta de la imagen guardada."""
    target_height = 760
    title_height = 56
    footer_height = 100
    padding = 20

    with Image.open(baseline.directory / "patch_selection_preview.png") as baseline_image:
        baseline_preview = baseline_image.convert("RGB")
    with Image.open(smart.directory / "patch_selection_preview.png") as smart_image:
        smart_preview = smart_image.convert("RGB")

    baseline_preview = _resize_to_height(baseline_preview, target_height)
    smart_preview = _resize_to_height(smart_preview, target_height)

    width = baseline_preview.width + smart_preview.width + padding * 3
    height = title_height + target_height + footer_height + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    baseline_x = padding
    smart_x = baseline_x + baseline_preview.width + padding
    image_y = title_height
    canvas.paste(baseline_preview, (baseline_x, image_y))
    canvas.paste(smart_preview, (smart_x, image_y))
    draw.text((baseline_x, 20), _selector_title(baseline), fill=(20, 20, 20), font=font)
    draw.text((smart_x, 20), _selector_title(smart), fill=(20, 20, 20), font=font)

    footer_y = image_y + target_height + padding
    for index, line in enumerate(_preview_footer_lines(overlap_metrics, feature_metrics)):
        draw.text((padding, footer_y + index * 22), line, fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def save_selected_only_comparison_preview(
    baseline: SelectorRun,
    smart: SelectorRun,
    output_path: Path,
    *,
    baseline_records: list[dict[str, object]],
    smart_records: list[dict[str, object]],
    overlap_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
) -> Path:
    """Crea una preview de ambas WSI mostrando solo los patches seleccionados."""
    target_height = 760
    title_height = 56
    footer_height = 100
    padding = 20

    baseline_preview = _draw_selected_boxes_on_thumbnail(
        _load_clean_thumbnail(baseline, baseline_records),
        baseline,
        baseline_records,
    )
    smart_preview = _draw_selected_boxes_on_thumbnail(
        _load_clean_thumbnail(smart, smart_records),
        smart,
        smart_records,
    )
    baseline_preview = _resize_to_height(baseline_preview, target_height)
    smart_preview = _resize_to_height(smart_preview, target_height)

    width = baseline_preview.width + smart_preview.width + padding * 3
    height = title_height + target_height + footer_height + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    baseline_x = padding
    smart_x = baseline_x + baseline_preview.width + padding
    image_y = title_height
    canvas.paste(baseline_preview, (baseline_x, image_y))
    canvas.paste(smart_preview, (smart_x, image_y))
    draw.text((baseline_x, 20), _selector_title(baseline), fill=(20, 20, 20), font=font)
    draw.text((smart_x, 20), _selector_title(smart), fill=(20, 20, 20), font=font)

    footer_y = image_y + target_height + padding
    for index, line in enumerate(_preview_footer_lines(overlap_metrics, feature_metrics)):
        draw.text((padding, footer_y + index * 22), line, fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def compare_patch_selectors(config: ComparisonConfig) -> dict[str, Any]:
    """
    ***
    * config: Rutas y opciones de la comparación.
    ***
    Valida que ambos métodos compartan configuración y pool, recomputa features sobre
    los patches seleccionados y calcula métricas técnicas y espaciales. Guarda CSV,
    JSON y previews, y retorna el resumen de la comparación.
    """
    if config.feature_size <= 0:
        raise ValueError("--feature-size debe ser mayor que cero.")

    root_dir = config.root_dir.resolve()
    baseline_dir = _resolve_path(config.baseline_dir, root_dir)
    smart_dir = _resolve_path(config.smart_dir, root_dir)
    output_dir = _resolve_path(config.output_dir, root_dir)
    _prepare_output_dir(output_dir=output_dir, root_dir=root_dir, overwrite=config.overwrite)

    baseline = load_selector_run("baseline", baseline_dir)
    smart = load_selector_run("smart", smart_dir)
    shared_config, validation_warnings = validate_shared_config(baseline, smart)
    if config.require_exact_candidate_pool:
        hash_matches = shared_config["candidate_pool_hash_exact"]["matches"]
        coordinates_match = shared_config["candidate_pool_coordinates_exact"]["matches"]
        if not hash_matches or not coordinates_match:
            raise ValueError(
                "Falló la validación exacta del pool común; la comparación no es válida."
            )

    baseline_records = _selected_records(baseline)
    smart_records = _selected_records(smart)
    overlap_metrics, overlap_rows = compute_overlap_metrics(baseline_records, smart_records)

    selected_patch_rows = []
    selected_patch_rows.extend(
        recompute_selected_patch_features(
            baseline,
            baseline_records,
            feature_size=config.feature_size,
            recompute=config.recompute_selected_features,
        )
    )
    selected_patch_rows.extend(
        recompute_selected_patch_features(
            smart,
            smart_records,
            feature_size=config.feature_size,
            recompute=config.recompute_selected_features,
        )
    )

    feature_names = [
        "tissue_ratio_recomputed",
        "nuclear_signal_recomputed",
        "nuclear_signal_rgb_recomputed",
        "nuclear_signal_hed_recomputed",
        "visual_entropy_recomputed",
        "blur_score_recomputed",
        "artifact_penalty_recomputed",
    ]
    feature_metrics = {
        "baseline": _method_feature_stats(
            [row for row in selected_patch_rows if row["method"] == "baseline"],
            feature_names,
        ),
        "smart": _method_feature_stats(
            [row for row in selected_patch_rows if row["method"] == "smart"],
            feature_names,
        ),
    }
    spatial_metrics = {
        "baseline": compute_spatial_metrics(baseline_records, baseline.summary),
        "smart": compute_spatial_metrics(smart_records, smart.summary),
    }
    optional_selector_metrics = compute_optional_selector_metrics(baseline, smart)
    runtime_metrics = {
        "baseline_runtime_seconds": baseline.summary.get("runtime_seconds"),
        "smart_runtime_seconds": smart.summary.get("runtime_seconds"),
    }
    metrics_rows = build_comparison_metrics_rows(
        baseline=baseline,
        smart=smart,
        overlap_metrics=overlap_metrics,
        feature_metrics=feature_metrics,
        spatial_metrics=spatial_metrics,
        optional_selector_metrics=optional_selector_metrics,
    )

    comparison_metrics_path = output_dir / "comparison_metrics.csv"
    selected_overlap_path = output_dir / "selected_overlap.csv"
    selected_patches_path = output_dir / "comparison_selected_patches.csv"
    preview_path = output_dir / "comparison_preview.png"
    selected_only_preview_path = output_dir / "comparison_preview_selected_only.png"
    summary_path = output_dir / "comparison_summary.json"

    _write_csv(metrics_rows, comparison_metrics_path, COMPARISON_METRICS_FIELDS)
    _write_csv(overlap_rows, selected_overlap_path, SELECTED_OVERLAP_FIELDS)
    _write_csv(selected_patch_rows, selected_patches_path, COMPARISON_SELECTED_PATCH_FIELDS)
    save_comparison_preview(
        baseline,
        smart,
        preview_path,
        overlap_metrics=overlap_metrics,
        feature_metrics=feature_metrics,
    )
    save_selected_only_comparison_preview(
        baseline,
        smart,
        selected_only_preview_path,
        baseline_records=baseline_records,
        smart_records=smart_records,
        overlap_metrics=overlap_metrics,
        feature_metrics=feature_metrics,
    )
    summary = {
        "baseline_dir": str(baseline_dir),
        "smart_dir": str(smart_dir),
        "output_dir": str(output_dir),
        "created_at": _utc_now_iso(),
        "validation_warnings": validation_warnings,
        "shared_config": shared_config,
        "baseline_summary": baseline.summary,
        "smart_summary": smart.summary,
        "overlap_metrics": overlap_metrics,
        "feature_metrics": feature_metrics,
        "spatial_metrics": spatial_metrics,
        "optional_selector_metrics": optional_selector_metrics,
        "runtime_metrics": runtime_metrics,
        "outputs": {
            "comparison_summary_json": str(summary_path),
            "comparison_metrics_csv": str(comparison_metrics_path),
            "selected_overlap_csv": str(selected_overlap_path),
            "comparison_selected_patches_csv": str(selected_patches_path),
            "comparison_preview_png": str(preview_path),
            "comparison_preview_selected_only_png": str(selected_only_preview_path),
        },
    }
    _write_json(summary, summary_path)
    return summary
