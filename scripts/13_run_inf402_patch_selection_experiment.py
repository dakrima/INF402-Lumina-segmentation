#!/usr/bin/env python
"""Run the nine-WSI INF402 baseline-vs-v4.1 experiment with bounded overlap."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import psutil

from src.selection.comparison import ComparisonConfig, compare_patch_selectors
from src.selection.embedding_scoring import (
    EmbeddingExtractorConfig,
    build_embedding_extractor,
    embedding_extractor_load_count,
)
from src.selection.tiatoolbox_baseline import (
    BASELINE_SELECTOR_NAME,
    BaselineSelectionConfig,
    TiatoolboxCandidate,
    candidate_pool_hash,
    generate_shared_candidate_pool,
    run_baseline_selection,
    write_shared_candidate_manifest,
)
from src.selection.v4_1_medical_embedding_assisted import (
    V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME,
    V41MedicalEmbeddingAssistedConfig,
    run_v4_1_medical_embedding_assisted_selection,
)


DEFAULT_WSI_DIR = Path("/Users/davidkripper/demoCasesMvpFeria/CasosINF402")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs/patch_selection/inf402_n9"
DEFAULT_UNI_MODEL = Path("/Users/davidkripper/models/uni/pytorch_model.bin")
PILOT_CASE_IDS = ("TCGA-E2-A1L7", "TCGA-C8-A26Y")
MIB = 1024 * 1024

CORE_METRICS = {
    "mean_tissue_ratio_recomputed": "Proporción media de tejido",
    "mean_nuclear_signal_hed_recomputed": "Señal HED media",
    "mean_visual_entropy_recomputed": "Entropía visual media",
    "mean_blur_score_recomputed": "Nitidez media",
    "mean_artifact_penalty_recomputed": "Penalización media por artefactos",
    "mean_pairwise_distance": "Distancia espacial media",
    "mean_nearest_neighbor_distance": "Distancia media al vecino más cercano",
    "spatial_coverage_ratio_approx": "Cobertura espacial aproximada",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded-concurrency INF402 patch-selection experiment.",
    )
    parser.add_argument("--wsi-dir", type=Path, default=DEFAULT_WSI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--uni-model-path", type=Path, default=DEFAULT_UNI_MODEL)
    parser.add_argument("--expected-count", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def case_id_from_path(path: Path) -> str:
    parts = path.name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive TCGA case id from {path.name}.")
    return "-".join(parts[:3])


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class MemoryMonitor:
    """Sample process memory and system swap while cases are active."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self.active_cases: set[str] = set()
        self.case_stats: dict[str, dict[str, float]] = {}
        self.samples = 0
        self.start_swap_mb = psutil.swap_memory().used / MIB
        self.current_swap_mb = self.start_swap_mb
        self.peak_swap_mb = self.start_swap_mb
        self.peak_rss_mb = self.process.memory_info().rss / MIB
        self.min_available_mb = psutil.virtual_memory().available / MIB
        self.gate_swap_baseline_mb = self.start_swap_mb
        self.gate_peak_swap_mb = self.start_swap_mb
        self.gate_min_available_mb = self.min_available_mb

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds * 2)
        self._sample()

    def activate(self, case_id: str) -> None:
        with self.lock:
            self.active_cases.add(case_id)
            self.case_stats[case_id] = {
                "start_swap_mb": self.current_swap_mb,
                "peak_swap_mb": self.current_swap_mb,
                "peak_rss_mb": self.peak_rss_mb,
                "min_available_mb": psutil.virtual_memory().available / MIB,
            }

    def deactivate(self, case_id: str) -> dict[str, float]:
        self._sample()
        with self.lock:
            self.active_cases.discard(case_id)
            stats = dict(self.case_stats.get(case_id, {}))
            stats["end_swap_mb"] = self.current_swap_mb
            stats["swap_delta_mb"] = max(
                0.0,
                stats.get("peak_swap_mb", self.current_swap_mb)
                - stats.get("start_swap_mb", self.current_swap_mb),
            )
            return {key: round(value, 3) for key, value in stats.items()}

    def reset_gate(self) -> None:
        self._sample()
        with self.lock:
            self.gate_swap_baseline_mb = self.current_swap_mb
            self.gate_peak_swap_mb = self.current_swap_mb
            self.gate_min_available_mb = psutil.virtual_memory().available / MIB

    def gate_summary(self) -> dict[str, float | bool]:
        with self.lock:
            swap_delta = max(0.0, self.gate_peak_swap_mb - self.gate_swap_baseline_mb)
            return {
                "swap_baseline_mb": round(self.gate_swap_baseline_mb, 3),
                "peak_swap_mb": round(self.gate_peak_swap_mb, 3),
                "swap_delta_mb": round(swap_delta, 3),
                "min_available_mb": round(self.gate_min_available_mb, 3),
                "memory_pressure": swap_delta > 256.0 or self.gate_min_available_mb < 512.0,
            }

    def summary(self) -> dict[str, float | int]:
        with self.lock:
            return {
                "samples": self.samples,
                "start_swap_mb": round(self.start_swap_mb, 3),
                "end_swap_mb": round(self.current_swap_mb, 3),
                "peak_swap_mb": round(self.peak_swap_mb, 3),
                "swap_delta_peak_mb": round(max(0.0, self.peak_swap_mb - self.start_swap_mb), 3),
                "peak_rss_mb": round(self.peak_rss_mb, 3),
                "min_available_mb": round(self.min_available_mb, 3),
            }

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        rss_mb = self.process.memory_info().rss / MIB
        swap_mb = psutil.swap_memory().used / MIB
        available_mb = psutil.virtual_memory().available / MIB
        with self.lock:
            self.samples += 1
            self.current_swap_mb = swap_mb
            self.peak_swap_mb = max(self.peak_swap_mb, swap_mb)
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            self.min_available_mb = min(self.min_available_mb, available_mb)
            self.gate_peak_swap_mb = max(self.gate_peak_swap_mb, swap_mb)
            self.gate_min_available_mb = min(self.gate_min_available_mb, available_mb)
            for case_id in self.active_cases:
                stats = self.case_stats[case_id]
                stats["peak_swap_mb"] = max(stats["peak_swap_mb"], swap_mb)
                stats["peak_rss_mb"] = max(stats["peak_rss_mb"], rss_mb)
                stats["min_available_mb"] = min(stats["min_available_mb"], available_mb)


def baseline_config(wsi_path: Path, output_dir: Path) -> BaselineSelectionConfig:
    return BaselineSelectionConfig(
        wsi_path=wsi_path,
        output_dir=output_dir,
        root_dir=ROOT_DIR,
        selector=BASELINE_SELECTOR_NAME,
        patch_size=1024,
        stride=1024,
        max_patches=16,
        min_tissue_ratio=0.20,
        seed=42,
        overwrite=True,
    )


def v41_config(
    wsi_path: Path,
    output_dir: Path,
    uni_model_path: Path,
    batch_size: int,
) -> V41MedicalEmbeddingAssistedConfig:
    return V41MedicalEmbeddingAssistedConfig(
        wsi_path=wsi_path,
        output_dir=output_dir,
        root_dir=ROOT_DIR,
        selector=V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME,
        patch_size=1024,
        stride=1024,
        max_patches=16,
        min_tissue_ratio=0.20,
        seed=42,
        overwrite=True,
        max_candidates_to_score=1000,
        feature_size=512,
        lambda_spatial=0.15,
        quota_grid="4x4",
        quota_min_score_quantile=0.20,
        feature_diversity_weight=0.10,
        redundancy_penalty_weight=0.10,
        min_quality_score=0.15,
        embedding_backend="uni",
        embedding_model_name="UNI",
        embedding_model_path=uni_model_path,
        embedding_device="cpu",
        embedding_batch_size=batch_size,
        embedding_num_workers=2,
        cache_embeddings=True,
        reuse_embedding_cache=True,
        embedding_distance_metric="cosine",
        embedding_diversity_weight=0.08,
        embedding_redundancy_weight=0.08,
        embedding_cluster_count=8,
        cluster_balance_weight=0.05,
        representative_cluster_weight=0.05,
        medical_min_quality_score=0.50,
        medical_min_utility_score=0.45,
        min_score_v3_base_quantile=0.80,
        medical_top_quantile=0.20,
        medical_artifact_max=0.12,
        medical_rerank_mode="top_v3_then_embedding",
    )


def _is_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in message or "cannot allocate" in message


def run_case(
    wsi_path: Path,
    *,
    output_root: Path,
    uni_model_path: Path,
    embedding_extractor: object,
    batch_size: int,
    monitor: MemoryMonitor,
) -> dict[str, Any]:
    case_id = case_id_from_path(wsi_path)
    case_root = output_root / case_id
    baseline_dir = case_root / "baseline"
    smart_dir = case_root / "v4_1"
    comparison_dir = case_root / "comparison"
    case_started = time.perf_counter()
    monitor.activate(case_id)
    status: dict[str, Any] = {
        "case_id": case_id,
        "wsi_path": str(wsi_path),
        "status": "running",
        "embedding_batch_size": batch_size,
    }
    print(f"[START] {case_id} (UNI batch {batch_size})", flush=True)
    pool = None
    try:
        base_config = baseline_config(wsi_path, baseline_dir)
        pool = generate_shared_candidate_pool(base_config, case_id=case_id)
        shared_csv, shared_json = write_shared_candidate_manifest(pool, case_root)
        base_summary = run_baseline_selection(base_config, shared_pool=pool)
        pool.release_extractor()

        smart_config = v41_config(wsi_path, smart_dir, uni_model_path, batch_size)
        retried_with_batch_8 = False
        try:
            smart_summary = run_v4_1_medical_embedding_assisted_selection(
                smart_config,
                shared_pool=pool,
                embedding_extractor=embedding_extractor,
            )
        except Exception as exc:
            if batch_size <= 8 or not _is_memory_error(exc):
                raise
            retried_with_batch_8 = True
            gc.collect()
            smart_config = v41_config(wsi_path, smart_dir, uni_model_path, 8)
            smart_summary = run_v4_1_medical_embedding_assisted_selection(
                smart_config,
                shared_pool=pool,
                embedding_extractor=embedding_extractor,
            )

        if base_summary.get("candidate_pool_hash") != smart_summary.get("candidate_pool_hash"):
            raise RuntimeError("Baseline and v4.1 candidate hashes differ.")
        if base_summary.get("candidate_pool_count") != smart_summary.get("candidate_pool_count"):
            raise RuntimeError("Baseline and v4.1 candidate counts differ.")

        comparison_started = time.perf_counter()
        comparison_summary = compare_patch_selectors(
            ComparisonConfig(
                baseline_dir=baseline_dir,
                smart_dir=smart_dir,
                output_dir=comparison_dir,
                root_dir=ROOT_DIR,
                feature_size=256,
                overwrite=True,
                recompute_selected_features=True,
                require_exact_candidate_pool=True,
            )
        )
        comparison_seconds = round(time.perf_counter() - comparison_started, 6)
        end_to_end_seconds = round(time.perf_counter() - case_started, 6)
        timings = {
            "candidate_generation_seconds": pool.candidate_generation_seconds,
            "baseline_selection_seconds": base_summary["baseline_selection_seconds"],
            "v41_feature_seconds": smart_summary["v41_feature_seconds"],
            "uni_embedding_seconds": smart_summary["uni_embedding_seconds"],
            "uni_lock_wait_seconds": smart_summary["uni_lock_wait_seconds"],
            "v41_rerank_seconds": smart_summary["v41_rerank_seconds"],
            "comparison_seconds": comparison_seconds,
            "end_to_end_seconds": end_to_end_seconds,
        }
        timings["active_compute_seconds_without_uni_wait"] = round(
            timings["candidate_generation_seconds"]
            + timings["baseline_selection_seconds"]
            + timings["v41_feature_seconds"]
            + max(0.0, timings["uni_embedding_seconds"] - timings["uni_lock_wait_seconds"])
            + timings["v41_rerank_seconds"]
            + timings["comparison_seconds"],
            6,
        )
        status.update(
            {
                "status": "completed",
                "embedding_batch_size": smart_config.embedding_batch_size,
                "retried_with_batch_8": retried_with_batch_8,
                "candidate_pool_hash": pool.candidate_pool_hash,
                "candidate_pool_count": len(pool.candidates),
                "num_selected_baseline": base_summary.get("num_selected"),
                "num_selected_v41": smart_summary.get("num_selected"),
                "objective_power": smart_summary.get("objective_power"),
                "mpp_x": smart_summary.get("mpp_x"),
                "mpp_y": smart_summary.get("mpp_y"),
                "timings": timings,
                "outputs": {
                    "shared_candidates_csv": str(shared_csv),
                    "shared_candidates_json": str(shared_json),
                    "baseline_dir": str(baseline_dir),
                    "v41_dir": str(smart_dir),
                    "comparison_dir": str(comparison_dir),
                    "comparison_summary": comparison_summary["outputs"]["comparison_summary_json"],
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - each WSI must fail independently
        status.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "end_to_end_seconds": round(time.perf_counter() - case_started, 6),
            }
        )
    finally:
        if pool is not None:
            pool.release_extractor()
        status["memory"] = monitor.deactivate(case_id)
        _atomic_json(status, case_root / "run_status.json")
        elapsed = status.get("timings", {}).get("end_to_end_seconds", status.get("end_to_end_seconds"))
        print(f"[{status['status'].upper()}] {case_id} ({elapsed}s)", flush=True)
    return status


def _read_metric_rows(path: Path) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            values: dict[str, float | None] = {}
            for field in ("baseline_value", "smart_value", "delta_smart_minus_baseline"):
                try:
                    values[field] = float(row[field]) if row.get(field) not in ("", None) else None
                except ValueError:
                    values[field] = None
            metrics[row["metric"]] = values
    return metrics


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None}
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else (values[0], values[0], values[0])
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
    }


def write_aggregate_outputs(
    case_statuses: list[dict[str, Any]],
    *,
    output_root: Path,
    scheduler: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, str]:
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    successful = [status for status in case_statuses if status.get("status") == "completed"]
    per_wsi_rows: list[dict[str, Any]] = []
    aggregate_values: dict[str, dict[str, list[float]]] = {
        metric: {"baseline": [], "v41": [], "delta": []}
        for metric in CORE_METRICS
    }
    for status in successful:
        comparison_dir = Path(status["outputs"]["comparison_dir"])
        metrics = _read_metric_rows(comparison_dir / "comparison_metrics.csv")
        row: dict[str, Any] = {
            "case_id": status["case_id"],
            "wsi_path": status["wsi_path"],
            "objective_power": status.get("objective_power"),
            "mpp_x": status.get("mpp_x"),
            "mpp_y": status.get("mpp_y"),
            "candidate_pool_hash": status.get("candidate_pool_hash"),
            "candidate_pool_count": status.get("candidate_pool_count"),
            "num_selected_baseline": status.get("num_selected_baseline"),
            "num_selected_v41": status.get("num_selected_v41"),
            "embedding_batch_size": status.get("embedding_batch_size"),
            **status.get("timings", {}),
            "peak_rss_mb": status.get("memory", {}).get("peak_rss_mb"),
            "swap_delta_mb": status.get("memory", {}).get("swap_delta_mb"),
        }
        for metric in CORE_METRICS:
            values = metrics.get(metric, {})
            baseline_value = values.get("baseline_value")
            smart_value = values.get("smart_value")
            delta = values.get("delta_smart_minus_baseline")
            row[f"{metric}_baseline"] = baseline_value
            row[f"{metric}_v41"] = smart_value
            row[f"{metric}_delta"] = delta
            if baseline_value is not None and smart_value is not None and delta is not None:
                aggregate_values[metric]["baseline"].append(float(baseline_value))
                aggregate_values[metric]["v41"].append(float(smart_value))
                aggregate_values[metric]["delta"].append(float(delta))
        per_wsi_rows.append(row)

    per_wsi_path = aggregate_dir / "per_wsi_metrics.csv"
    if per_wsi_rows:
        with per_wsi_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(per_wsi_rows[0]))
            writer.writeheader()
            writer.writerows(per_wsi_rows)
    else:
        per_wsi_path.write_text("case_id\n", encoding="utf-8")

    aggregate_rows: list[dict[str, Any]] = []
    for metric, label in CORE_METRICS.items():
        distributions = {
            key: _distribution(values)
            for key, values in aggregate_values[metric].items()
        }
        aggregate_rows.append(
            {
                "metric": metric,
                "label": label,
                **{f"baseline_{key}": value for key, value in distributions["baseline"].items()},
                **{f"v41_{key}": value for key, value in distributions["v41"].items()},
                **{f"delta_{key}": value for key, value in distributions["delta"].items()},
            }
        )
    aggregate_metrics_path = aggregate_dir / "aggregate_metrics.csv"
    with aggregate_metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    summary_path = aggregate_dir / "aggregate_summary.json"
    _atomic_json(
        {
            "status": "completed" if len(successful) == len(case_statuses) else "completed_with_failures",
            "num_cases_requested": len(case_statuses),
            "num_cases_completed": len(successful),
            "num_cases_failed": len(case_statuses) - len(successful),
            "failed_cases": [
                {"case_id": item["case_id"], "error": item.get("error")}
                for item in case_statuses
                if item.get("status") != "completed"
            ],
            "scheduler": scheduler,
            "memory": memory,
            "clinical_warning": "Technical patch selection only; not diagnosis or clinical validation.",
        },
        summary_path,
    )

    paper_path = aggregate_dir / "paper_results.md"
    lines = [
        "# Resultados técnicos de selección de patches",
        "",
        f"Se completaron {len(successful)} de {len(case_statuses)} WSI.",
        "La comparación es técnica y no constituye validación clínica.",
        "",
        "| Métrica | Baseline (media ± DE) | v4.1 (media ± DE) | Diferencia pareada media |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        if row["baseline_mean"] is None:
            continue
        lines.append(
            f"| {row['label']} | {row['baseline_mean']:.4f} ± {row['baseline_sd']:.4f} "
            f"| {row['v41_mean']:.4f} ± {row['v41_sd']:.4f} | {row['delta_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Los tiempos separan la generación común de candidatos del costo incremental de cada selector.",
            "Las WSI se procesaron en nivel 0; las diferencias de MPP deben considerarse una limitación.",
        ]
    )
    paper_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "per_wsi_metrics_csv": str(per_wsi_path),
        "aggregate_metrics_csv": str(aggregate_metrics_path),
        "aggregate_summary_json": str(summary_path),
        "paper_results_md": str(paper_path),
    }


def self_check() -> None:
    first = TiatoolboxCandidate("a", 0, 10, 20, 1024, 0)
    second = TiatoolboxCandidate("b", 1, 30, 40, 1024, 1)
    expected = candidate_pool_hash("TCGA-X-Y", [first, second])
    assert expected == candidate_pool_hash("TCGA-X-Y", [second, first])
    changed = TiatoolboxCandidate("b", 1, 31, 40, 1024, 1)
    assert expected != candidate_pool_hash("TCGA-X-Y", [first, changed])
    assert expected != candidate_pool_hash("TCGA-X-Z", [first, second])
    print("[OK] Candidate pool hash self-check passed.")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    wsi_dir = args.wsi_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser()
    if not output_root.is_absolute():
        output_root = (ROOT_DIR / output_root).resolve()
    uni_model_path = args.uni_model_path.expanduser().resolve()
    paths = sorted(wsi_dir.glob("*.svs"))
    if len(paths) != args.expected_count:
        print(f"[FAIL] Expected {args.expected_count} WSI files, found {len(paths)} in {wsi_dir}.")
        return 1
    if not uni_model_path.exists():
        print(f"[FAIL] UNI model does not exist: {uni_model_path}")
        return 1
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_started = time.perf_counter()

    by_case = {case_id_from_path(path): path for path in paths}
    missing_pilot = [case_id for case_id in PILOT_CASE_IDS if case_id not in by_case]
    if missing_pilot:
        print(f"[FAIL] Missing pilot cases: {', '.join(missing_pilot)}")
        return 1
    pilot_paths = [by_case[case_id] for case_id in PILOT_CASE_IDS]
    remaining_paths = [path for path in paths if path not in pilot_paths]

    monitor = MemoryMonitor(interval_seconds=2.0)
    monitor.start()
    model_load_started = time.perf_counter()
    embedding_extractor = build_embedding_extractor(
        EmbeddingExtractorConfig(
            embedding_backend="uni",
            embedding_model_name="UNI",
            embedding_model_path=uni_model_path,
            embedding_device="cpu",
            embedding_batch_size=16,
            embedding_num_workers=2,
            embedding_distance_metric="cosine",
        )
    )
    model_load_seconds = round(time.perf_counter() - model_load_started, 6)
    monitor.reset_gate()

    case_statuses: list[dict[str, Any]] = []
    pilot_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="inf402-wsi") as executor:
        pilot_futures = [
            executor.submit(
                run_case,
                path,
                output_root=output_root,
                uni_model_path=uni_model_path,
                embedding_extractor=embedding_extractor,
                batch_size=16,
                monitor=monitor,
            )
            for path in pilot_paths
        ]
        for future in pilot_futures:
            case_statuses.append(future.result())
    pilot_wall_seconds = round(time.perf_counter() - pilot_started, 6)
    pilot_gate = monitor.gate_summary()
    pilot_success = [item for item in case_statuses if item.get("status") == "completed"]
    sequential_counterfactual = sum(
        item["timings"]["active_compute_seconds_without_uni_wait"]
        for item in pilot_success
    )
    speedup_fraction = (
        max(0.0, 1.0 - pilot_wall_seconds / sequential_counterfactual)
        if sequential_counterfactual > 0
        else 0.0
    )
    continue_concurrent = (
        len(pilot_success) == 2
        and speedup_fraction >= 0.05
        and not pilot_gate["memory_pressure"]
        and embedding_extractor_load_count() == 1
    )
    batch_size = 8 if pilot_gate["memory_pressure"] else 16
    scheduler: dict[str, Any] = {
        "pilot_case_ids": list(PILOT_CASE_IDS),
        "pilot_wall_seconds": pilot_wall_seconds,
        "pilot_sequential_counterfactual_seconds": round(sequential_counterfactual, 6),
        "pilot_speedup_fraction": round(speedup_fraction, 6),
        "pilot_memory": pilot_gate,
        "continue_concurrent": continue_concurrent,
        "remaining_batch_size": batch_size,
        "max_active_wsi": 2 if continue_concurrent else 1,
        "uni_model_load_seconds": model_load_seconds,
    }
    print(
        "[PILOT] "
        f"speedup={speedup_fraction:.1%}, swap_delta={pilot_gate['swap_delta_mb']} MiB, "
        f"min_available={pilot_gate['min_available_mb']} MiB, "
        f"continue_concurrent={continue_concurrent}",
        flush=True,
    )

    if continue_concurrent:
        monitor.reset_gate()
        queue = iter(remaining_paths)
        pending: dict[Future[dict[str, Any]], Path] = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="inf402-wsi") as executor:
            for _ in range(2):
                try:
                    path = next(queue)
                except StopIteration:
                    break
                pending[
                    executor.submit(
                        run_case,
                        path,
                        output_root=output_root,
                        uni_model_path=uni_model_path,
                        embedding_extractor=embedding_extractor,
                        batch_size=batch_size,
                        monitor=monitor,
                    )
                ] = path
            sequential_remainder: list[Path] = []
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    case_statuses.append(future.result())
                pressure = bool(monitor.gate_summary()["memory_pressure"])
                if pressure:
                    sequential_remainder.extend(list(queue))
                    continue
                while len(pending) < 2:
                    try:
                        path = next(queue)
                    except StopIteration:
                        break
                    pending[
                        executor.submit(
                            run_case,
                            path,
                            output_root=output_root,
                            uni_model_path=uni_model_path,
                            embedding_extractor=embedding_extractor,
                            batch_size=batch_size,
                            monitor=monitor,
                        )
                    ] = path
        if sequential_remainder:
            scheduler["concurrency_disabled_after_pressure"] = True
            scheduler["remaining_batch_size"] = 8
            scheduler["max_active_wsi"] = 1
            for path in sequential_remainder:
                case_statuses.append(
                    run_case(
                        path,
                        output_root=output_root,
                        uni_model_path=uni_model_path,
                        embedding_extractor=embedding_extractor,
                        batch_size=8,
                        monitor=monitor,
                    )
                )
    else:
        for path in remaining_paths:
            case_statuses.append(
                run_case(
                    path,
                    output_root=output_root,
                    uni_model_path=uni_model_path,
                    embedding_extractor=embedding_extractor,
                    batch_size=batch_size,
                    monitor=monitor,
                )
            )

    monitor.stop()
    scheduler["uni_model_load_count"] = embedding_extractor_load_count()
    scheduler["total_wall_seconds"] = round(time.perf_counter() - experiment_started, 6)
    case_statuses.sort(key=lambda item: item["case_id"])
    memory_summary = monitor.summary()
    outputs = write_aggregate_outputs(
        case_statuses,
        output_root=output_root,
        scheduler=scheduler,
        memory=memory_summary,
    )
    run_status = {
        "status": "completed" if all(item.get("status") == "completed" for item in case_statuses) else "completed_with_failures",
        "wsi_dir": str(wsi_dir),
        "output_dir": str(output_root),
        "scheduler": scheduler,
        "memory": memory_summary,
        "cases": case_statuses,
        "aggregate_outputs": outputs,
    }
    _atomic_json(run_status, output_root / "run_status.json")
    completed = sum(item.get("status") == "completed" for item in case_statuses)
    print(f"[OK] Completed {completed}/{len(case_statuses)} WSI comparisons.")
    print(f"[OK] Aggregate summary: {outputs['aggregate_summary_json']}")
    return 0 if completed == len(case_statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
