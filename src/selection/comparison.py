"""Compare patch selector outputs without opening WSI or running models."""

from __future__ import annotations

import csv
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


CLINICAL_WARNING = (
    "Technical patch selection comparison only. Not for diagnosis, not RCB, "
    "not clinical validation."
)
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
    "visual_entropy_recomputed",
    "blur_score_recomputed",
    "artifact_penalty_recomputed",
    "score_raw",
    "score_final",
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
]


@dataclass(frozen=True)
class SelectorRun:
    """Loaded selector output directory."""

    label: str
    directory: Path
    summary: dict[str, Any]
    method_config: dict[str, Any]
    candidate_rows: list[dict[str, str]]
    selected_rows: list[dict[str, str]]


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for comparing two selector output directories."""

    baseline_dir: Path
    smart_dir: Path
    output_dir: Path
    root_dir: Path
    feature_size: int = 256
    overwrite: bool = False
    recompute_selected_features: bool = True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_path(path: Path, root_dir: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (root_dir / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_output_dir(output_dir: Path, root_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    if resolved_output.exists() and any(child.name != ".gitkeep" for child in resolved_output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {resolved_output}. "
                "Use --overwrite to regenerate this comparison."
            )
        dangerous_paths = {
            Path("/").resolve(),
            Path.home().resolve(),
            resolved_root,
            resolved_root / "data",
            resolved_root / "outputs",
        }
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite only clears output directories inside the repository.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validate_result_dir(result_dir: Path) -> None:
    if not result_dir.exists():
        raise FileNotFoundError(f"Selector output directory does not exist: {result_dir}")
    for file_name in REQUIRED_RESULT_FILES:
        path = result_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    for dir_name in REQUIRED_RESULT_DIRS:
        path = result_dir / dir_name
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Missing required directory: {path}")


def load_selector_run(label: str, directory: Path) -> SelectorRun:
    """Load and validate a selector output directory."""
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
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _candidate_key_from_xy(x_level0: object, y_level0: object) -> str:
    return f"x{int(float(x_level0))}_y{int(float(y_level0))}"


def _candidate_lookup_by_xy(candidate_rows: list[dict[str, str]]) -> dict[tuple[int, int], str]:
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


def validate_shared_config(baseline: SelectorRun, smart: SelectorRun) -> tuple[dict[str, Any], list[str]]:
    """Validate shared experimental settings and return warnings."""
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
                f"Shared field mismatch for {field_name}: "
                f"baseline={baseline_value!r}, smart={smart_value!r}."
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
            "Candidate pool keys differ between methods; comparison may not use the same pool."
        )
    return shared_config, warnings


def compute_overlap_metrics(
    baseline_records: list[dict[str, object]],
    smart_records: list[dict[str, object]],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
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
    result: dict[str, dict[str, float | None]] = {}
    for feature_name in feature_names:
        values = [
            value for value in (_to_float(row.get(feature_name)) for row in rows)
            if value is not None
        ]
        result[feature_name] = _stats(values)
    return result


def recompute_selected_patch_features(
    run: SelectorRun,
    records: list[dict[str, object]],
    *,
    feature_size: int,
    recompute: bool,
) -> list[dict[str, object]]:
    """Build per-selected-patch comparison rows, reading one PNG at a time."""
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
                raise FileNotFoundError(f"Missing selected patch PNG: {patch_path}")
            with Image.open(patch_path) as image:
                features = compute_patch_features(
                    rgb_patch=image.convert("RGB"),
                    feature_size=feature_size,
                )
            output_row.update(
                {
                    "tissue_ratio_recomputed": f"{features['tissue_ratio']:.6f}",
                    "nuclear_signal_recomputed": f"{features['nuclear_signal']:.6f}",
                    "visual_entropy_recomputed": f"{features['visual_entropy']:.6f}",
                    "blur_score_recomputed": f"{features['blur_score']:.6f}",
                    "artifact_penalty_recomputed": f"{features['artifact_penalty']:.6f}",
                }
            )
        rows.append(output_row)
    return rows


def _pairwise_distances(records: list[dict[str, object]]) -> list[float]:
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
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    count_metrics = [
        "num_candidates_generated",
        "num_thumbnail_candidates_passing_mask",
        "num_candidate_rows_written",
        "num_candidates_evaluated",
        "num_candidates_scored",
        "num_selected",
    ]
    for metric in count_metrics:
        rows.append(
            _metric_row(
                metric,
                baseline.summary.get(metric, 0),
                smart.summary.get(metric, 0),
                higher_is_better=None,
                interpretation="Operational count; expected differences depend on selector design.",
            )
        )
    rows.append(
        _metric_row(
            "runtime_seconds",
            baseline.summary.get("runtime_seconds"),
            smart.summary.get("runtime_seconds"),
            higher_is_better=False,
            interpretation="Lower runtime is operationally preferable.",
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
                interpretation="Overlap is descriptive; higher is not necessarily better.",
            )
        )

    feature_specs = [
        ("tissue_ratio_recomputed", True, "Higher tissue fraction may indicate less empty background."),
        ("nuclear_signal_recomputed", True, "Higher proxy signal may indicate more hematoxylin-rich regions."),
        ("visual_entropy_recomputed", True, "Higher entropy may indicate richer visual variation."),
        ("blur_score_recomputed", True, "Higher gradient variance may indicate sharper texture."),
        ("artifact_penalty_recomputed", False, "Lower artifact penalty is preferable."),
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
                interpretation="Higher value suggests broader spatial dispersion, not clinical superiority.",
            )
        )
    return rows


def build_interpretation(
    feature_metrics: dict[str, Any],
    spatial_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_features = feature_metrics["baseline"]
    smart_features = feature_metrics["smart"]
    smart_tissue = smart_features["tissue_ratio_recomputed"]["mean"]
    baseline_tissue = baseline_features["tissue_ratio_recomputed"]["mean"]
    smart_nuclear = smart_features["nuclear_signal_recomputed"]["mean"]
    baseline_nuclear = baseline_features["nuclear_signal_recomputed"]["mean"]
    smart_artifact = smart_features["artifact_penalty_recomputed"]["mean"]
    baseline_artifact = baseline_features["artifact_penalty_recomputed"]["mean"]
    smart_nn = spatial_metrics["smart"].get("mean_nearest_neighbor_distance")
    baseline_nn = spatial_metrics["baseline"].get("mean_nearest_neighbor_distance")

    notes = [
        "Metrics are heuristic and technical; they do not imply diagnosis or clinical validation.",
        "Recomputed features use only selected PNG patches and feature_size, not the full WSI.",
    ]
    if smart_nuclear is not None and baseline_nuclear is not None and smart_nuclear > baseline_nuclear:
        notes.append("Smart selector chose patches with higher mean nuclear-signal proxy.")
    if smart_artifact is not None and baseline_artifact is not None and smart_artifact < baseline_artifact:
        notes.append("Smart selector chose patches with lower mean artifact penalty.")
    return {
        "smart_selects_more_tissue_dense_patches": bool(
            smart_tissue is not None and baseline_tissue is not None and smart_tissue > baseline_tissue
        ),
        "smart_has_higher_nuclear_signal": bool(
            smart_nuclear is not None and baseline_nuclear is not None and smart_nuclear > baseline_nuclear
        ),
        "smart_has_lower_artifact_penalty": bool(
            smart_artifact is not None and baseline_artifact is not None and smart_artifact < baseline_artifact
        ),
        "smart_has_lower_or_higher_spatial_diversity": (
            "higher"
            if smart_nn is not None and baseline_nn is not None and smart_nn > baseline_nn
            else "lower_or_equal"
            if smart_nn is not None and baseline_nn is not None
            else "unavailable"
        ),
        "notes": notes,
    }


def save_comparison_preview(
    baseline: SelectorRun,
    smart: SelectorRun,
    output_path: Path,
    *,
    overlap_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
) -> Path:
    """Create a lightweight side-by-side preview from existing selector previews."""
    target_height = 760
    title_height = 56
    footer_height = 100
    padding = 20

    with Image.open(baseline.directory / "patch_selection_preview.png") as baseline_image:
        baseline_preview = baseline_image.convert("RGB")
    with Image.open(smart.directory / "patch_selection_preview.png") as smart_image:
        smart_preview = smart_image.convert("RGB")

    def resize_to_height(image: Image.Image, height: int) -> Image.Image:
        ratio = height / image.height
        width = max(1, int(round(image.width * ratio)))
        return image.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)

    baseline_preview = resize_to_height(baseline_preview, target_height)
    smart_preview = resize_to_height(smart_preview, target_height)

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
    draw.text((baseline_x, 20), "baseline_tiatoolbox", fill=(20, 20, 20), font=font)
    draw.text((smart_x, 20), "smart_tissue_nuclei_v1", fill=(20, 20, 20), font=font)

    baseline_features = feature_metrics["baseline"]
    smart_features = feature_metrics["smart"]
    footer_y = image_y + target_height + padding
    lines = [
        (
            f"selected baseline/smart: {overlap_metrics['num_selected_baseline']} / "
            f"{overlap_metrics['num_selected_smart']} | overlap: "
            f"{overlap_metrics['num_overlap_selected']} | jaccard: "
            f"{overlap_metrics['jaccard_selected']:.3f}"
        ),
        (
            "mean tissue: "
            f"{baseline_features['tissue_ratio_recomputed']['mean']:.3f} / "
            f"{smart_features['tissue_ratio_recomputed']['mean']:.3f} | "
            "mean nuclear: "
            f"{baseline_features['nuclear_signal_recomputed']['mean']:.3f} / "
            f"{smart_features['nuclear_signal_recomputed']['mean']:.3f} | "
            "mean artifact: "
            f"{baseline_features['artifact_penalty_recomputed']['mean']:.3f} / "
            f"{smart_features['artifact_penalty_recomputed']['mean']:.3f}"
        ),
    ]
    for index, line in enumerate(lines):
        draw.text((padding, footer_y + index * 22), line, fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def write_comparison_notes(
    output_path: Path,
    interpretation: dict[str, Any],
    warnings: list[str],
) -> Path:
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None"
    note_lines = "\n".join(f"- {note}" for note in interpretation["notes"])
    body = f"""# Patch Selector Comparison Notes

This comparison is technical only. It does not diagnose, calculate RCB, replace a pathologist, or validate clinical performance.

## Interpretation

- Smart selects more tissue-dense patches: {interpretation['smart_selects_more_tissue_dense_patches']}
- Smart has higher nuclear-signal proxy: {interpretation['smart_has_higher_nuclear_signal']}
- Smart has lower artifact penalty: {interpretation['smart_has_lower_artifact_penalty']}
- Spatial diversity direction: {interpretation['smart_has_lower_or_higher_spatial_diversity']}

## Notes

{note_lines}

## Configuration Warnings

{warning_lines}
"""
    output_path.write_text(body, encoding="utf-8")
    return output_path


def compare_patch_selectors(config: ComparisonConfig) -> dict[str, Any]:
    """Compare baseline and smart selector output directories."""
    if config.feature_size <= 0:
        raise ValueError("--feature-size must be positive.")

    root_dir = config.root_dir.resolve()
    baseline_dir = _resolve_path(config.baseline_dir, root_dir)
    smart_dir = _resolve_path(config.smart_dir, root_dir)
    output_dir = _resolve_path(config.output_dir, root_dir)
    _prepare_output_dir(output_dir=output_dir, root_dir=root_dir, overwrite=config.overwrite)

    baseline = load_selector_run("baseline", baseline_dir)
    smart = load_selector_run("smart", smart_dir)
    shared_config, validation_warnings = validate_shared_config(baseline, smart)

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
    runtime_metrics = {
        "baseline_runtime_seconds": baseline.summary.get("runtime_seconds"),
        "smart_runtime_seconds": smart.summary.get("runtime_seconds"),
    }
    interpretation = build_interpretation(
        feature_metrics=feature_metrics,
        spatial_metrics=spatial_metrics,
    )
    metrics_rows = build_comparison_metrics_rows(
        baseline=baseline,
        smart=smart,
        overlap_metrics=overlap_metrics,
        feature_metrics=feature_metrics,
        spatial_metrics=spatial_metrics,
    )

    comparison_metrics_path = output_dir / "comparison_metrics.csv"
    selected_overlap_path = output_dir / "selected_overlap.csv"
    selected_patches_path = output_dir / "comparison_selected_patches.csv"
    preview_path = output_dir / "comparison_preview.png"
    summary_path = output_dir / "comparison_summary.json"
    notes_path = output_dir / "comparison_notes.md"

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
    write_comparison_notes(notes_path, interpretation, validation_warnings)

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
        "runtime_metrics": runtime_metrics,
        "interpretation": interpretation,
        "outputs": {
            "comparison_summary_json": str(summary_path),
            "comparison_metrics_csv": str(comparison_metrics_path),
            "selected_overlap_csv": str(selected_overlap_path),
            "comparison_selected_patches_csv": str(selected_patches_path),
            "comparison_preview_png": str(preview_path),
            "comparison_notes_md": str(notes_path),
        },
        "clinical_warning": CLINICAL_WARNING,
    }
    _write_json(summary, summary_path)
    return summary
