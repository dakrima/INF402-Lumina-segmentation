"""Smart tissue/nuclei patch selector for INF402 Etapa 2."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.wsi_patch_extraction import (
    SUPPORTED_WSI_EXTENSIONS,
    TISSUE_MASK_METHOD,
    _import_openslide,
)
from src.selection.candidate_generation import PatchCandidate, generate_tissue_candidates
from src.selection.diversity import (
    assign_spatial_regions,
    greedy_select_with_spatial_penalty,
    select_with_spatial_quotas,
)
from src.selection.manifests import (
    CANDIDATE_METADATA_FIELDS,
    SELECTED_METADATA_FIELDS,
    utc_now_iso,
    write_csv_manifest,
    write_json_manifest,
)
from src.selection.quality_filters import (
    ARTIFACT_PENALTY_METHOD,
    BLUR_SCORE_METHOD,
    VISUAL_ENTROPY_METHOD,
    compute_patch_features,
    nuclear_signal_method_for_proxy,
)
from src.selection.scoring import DEFAULT_SMART_WEIGHTS, apply_feature_scores, normalize_feature
from src.selection.previews import save_wsi_patch_selection_preview
from src.selection.tiatoolbox_baseline import (
    CANDIDATE_METADATA_SEMANTICS,
    CANDIDATE_POOL,
    CLINICAL_WARNING,
    _base_slide_metadata,
    _prepare_output_dir,
    _resolve_output_dir,
)


SMART_SELECTOR_NAME = "smart_tissue_nuclei_v1"
SMART_V2_LIGHT_SELECTOR_NAME = "smart_tissue_nuclei_v2_light"
SUPPORTED_SMART_SELECTORS = {SMART_SELECTOR_NAME, SMART_V2_LIGHT_SELECTOR_NAME}
CANDIDATE_ORDERING = "thumbnail_filtered_seeded_shuffle_then_feature_score"
PREVIEW_SHOWS = "selected_or_scored_candidates"
FEATURE_DIVERSITY_FIELDS = [
    "tissue_ratio_norm",
    "nuclear_signal_norm",
    "visual_entropy_norm",
    "blur_score_norm",
    "artifact_penalty_norm",
    "x_norm",
    "y_norm",
    "thumbnail_tissue_ratio_norm",
]


@dataclass(frozen=True)
class SmartTissueNucleiConfig:
    """Configuration for the CPU-friendly smart patch selector."""

    wsi_path: Path
    output_dir: Path
    root_dir: Path
    selector: str = SMART_SELECTOR_NAME
    patch_size: int = 1024
    stride: int = 1024
    max_patches: int = 16
    min_tissue_ratio: float = 0.20
    seed: int = 42
    thumbnail_max_size: int = 2048
    overwrite: bool = False
    max_candidates_to_score: int = 300
    feature_size: int = 256
    lambda_spatial: float = 0.15
    min_distance_level0: int | None = None
    nuclear_proxy: str = "rgb_purple"
    spatial_strategy: str = "penalty"
    quota_grid: str = "4x4"
    quota_min_score_quantile: float = 0.25
    diversity_strategy: str = "none"
    feature_diversity_weight: float = 0.10


def _validate_config(config: SmartTissueNucleiConfig, wsi_path: Path) -> None:
    if config.selector not in SUPPORTED_SMART_SELECTORS:
        raise NotImplementedError(
            f"Selector '{config.selector}' todavía no está implementado. "
            "Esta etapa soporta: " + ", ".join(sorted(SUPPORTED_SMART_SELECTORS))
        )
    if config.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if config.stride <= 0:
        raise ValueError("--stride must be positive.")
    if config.max_patches <= 0:
        raise ValueError("--max-patches must be positive.")
    if not 0 <= config.min_tissue_ratio <= 1:
        raise ValueError("--min-tissue-ratio must be between 0 and 1.")
    if config.thumbnail_max_size <= 0:
        raise ValueError("--thumbnail-max-size must be positive.")
    if config.max_candidates_to_score < 0:
        raise ValueError("--max-candidates-to-score must be >= 0.")
    if config.feature_size <= 0:
        raise ValueError("--feature-size must be positive.")
    if config.lambda_spatial < 0:
        raise ValueError("--lambda-spatial must be >= 0.")
    if config.min_distance_level0 is not None and config.min_distance_level0 <= 0:
        raise ValueError("--min-distance-level0 must be positive when provided.")
    if config.nuclear_proxy not in {"rgb_purple", "hed_deconvolution"}:
        raise ValueError("--nuclear-proxy must be rgb_purple or hed_deconvolution.")
    if config.spatial_strategy not in {"penalty", "quotas"}:
        raise ValueError("--spatial-strategy must be penalty or quotas.")
    if not 0 <= config.quota_min_score_quantile <= 1:
        raise ValueError("--quota-min-score-quantile must be between 0 and 1.")
    if config.diversity_strategy not in {"none", "farthest_feature"}:
        raise ValueError("--diversity-strategy must be none or farthest_feature.")
    if config.feature_diversity_weight < 0:
        raise ValueError("--feature-diversity-weight must be >= 0.")
    if wsi_path.suffix.lower() not in SUPPORTED_WSI_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_WSI_EXTENSIONS))
        raise ValueError(f"Unsupported WSI extension '{wsi_path.suffix}'. Use one of: {allowed}.")
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")


def _format_float(value: object) -> str:
    return f"{float(value):.6f}"


def _mean_selected(selected_rows: list[dict[str, object]], field_name: str) -> float | None:
    values: list[float] = []
    for row in selected_rows:
        value = row.get(field_name)
        if value in ("", None):
            continue
        values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _candidate_pool_row(
    candidate: PatchCandidate,
    *,
    config: SmartTissueNucleiConfig,
    wsi_path: Path,
    slide_metadata: dict[str, Any],
) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "grid_index": candidate.grid_index,
        "x_level0": candidate.x_level0,
        "y_level0": candidate.y_level0,
        "patch_size": candidate.patch_size,
        "width": "",
        "height": "",
        "thumbnail_tissue_ratio": _format_float(candidate.thumbnail_tissue_ratio),
        "evaluated": False,
        "scored": False,
        "nuclear_proxy": config.nuclear_proxy,
        "tissue_ratio": "",
        "nuclear_signal": "",
        "visual_entropy": "",
        "blur_score": "",
        "artifact_penalty": "",
        "spatial_penalty": "",
        "feature_diversity_bonus": "",
        "score_raw": "",
        "score_final": "",
        "region_id": "",
        "region_row": "",
        "region_col": "",
        "quota_grid": config.quota_grid if config.spatial_strategy == "quotas" else "",
        "spatial_strategy": config.spatial_strategy,
        "diversity_strategy": config.diversity_strategy,
        "selected": False,
        "rank": "",
        "filename": "",
        "selection_method": config.selector,
        "seed": config.seed,
        "source_wsi_path": str(wsi_path),
        **slide_metadata,
    }


def _selected_row(candidate_row: dict[str, object]) -> dict[str, object]:
    patch_id = str(candidate_row["filename"]).removesuffix(".png")
    return {
        "patch_id": patch_id,
        "filename": candidate_row["filename"],
        "selected": True,
        "rank": candidate_row["rank"],
        "x_level0": candidate_row["x_level0"],
        "y_level0": candidate_row["y_level0"],
        "patch_size": candidate_row["patch_size"],
        "width": candidate_row["width"],
        "height": candidate_row["height"],
        "thumbnail_tissue_ratio": candidate_row["thumbnail_tissue_ratio"],
        "tissue_ratio": candidate_row["tissue_ratio"],
        "nuclear_signal": candidate_row["nuclear_signal"],
        "visual_entropy": candidate_row["visual_entropy"],
        "blur_score": candidate_row["blur_score"],
        "artifact_penalty": candidate_row["artifact_penalty"],
        "spatial_penalty": candidate_row["spatial_penalty"],
        "feature_diversity_bonus": candidate_row["feature_diversity_bonus"],
        "score_raw": candidate_row["score_raw"],
        "score_final": candidate_row["score_final"],
        "nuclear_proxy": candidate_row["nuclear_proxy"],
        "region_id": candidate_row["region_id"],
        "region_row": candidate_row["region_row"],
        "region_col": candidate_row["region_col"],
        "quota_grid": candidate_row["quota_grid"],
        "spatial_strategy": candidate_row["spatial_strategy"],
        "diversity_strategy": candidate_row["diversity_strategy"],
        "source_wsi_path": candidate_row["source_wsi_path"],
        "slide_width": candidate_row["slide_width"],
        "slide_height": candidate_row["slide_height"],
        "objective_power": candidate_row["objective_power"],
        "mpp_x": candidate_row["mpp_x"],
        "mpp_y": candidate_row["mpp_y"],
        "level_count": candidate_row["level_count"],
        "selection_method": candidate_row["selection_method"],
        "seed": candidate_row["seed"],
    }


def _method_config(
    config: SmartTissueNucleiConfig,
    *,
    min_distance_level0: int,
) -> dict[str, object]:
    return {
        "selector": config.selector,
        "candidate_pool": CANDIDATE_POOL,
        "candidate_metadata_semantics": CANDIDATE_METADATA_SEMANTICS,
        "candidate_ordering": CANDIDATE_ORDERING,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "seed": config.seed,
        "thumbnail_max_size": config.thumbnail_max_size,
        "feature_size": config.feature_size,
        "max_candidates_to_score": config.max_candidates_to_score,
        "weights": DEFAULT_SMART_WEIGHTS,
        "nuclear_proxy": config.nuclear_proxy,
        "nuclear_signal_method": nuclear_signal_method_for_proxy(config.nuclear_proxy),
        "spatial_strategy": config.spatial_strategy,
        "lambda_spatial": config.lambda_spatial,
        "min_distance_level0": min_distance_level0,
        "quota_grid": config.quota_grid,
        "quota_min_score_quantile": config.quota_min_score_quantile,
        "diversity_strategy": config.diversity_strategy,
        "feature_diversity_weight": config.feature_diversity_weight,
        "tissue_mask_method": TISSUE_MASK_METHOD,
        "visual_entropy_method": VISUAL_ENTROPY_METHOD,
        "blur_score_method": BLUR_SCORE_METHOD,
        "artifact_penalty_method": ARTIFACT_PENALTY_METHOD,
        "created_at": utc_now_iso(),
    }


def _select_candidates_to_score(
    candidates: list[PatchCandidate],
    *,
    seed: int,
    max_candidates_to_score: int,
) -> list[PatchCandidate]:
    ordered_candidates = list(candidates)
    random.Random(seed).shuffle(ordered_candidates)
    if max_candidates_to_score == 0:
        return ordered_candidates
    return ordered_candidates[:max_candidates_to_score]


def run_smart_tissue_nuclei_selection(config: SmartTissueNucleiConfig) -> dict[str, Any]:
    """Run smart_tissue_nuclei_v1 and write comparable selection outputs."""
    start_time = time.perf_counter()
    root_dir = config.root_dir.resolve()
    wsi_path = config.wsi_path.expanduser().resolve()
    output_dir = _resolve_output_dir(config.output_dir, root_dir=root_dir)
    min_distance_level0 = config.min_distance_level0 or config.patch_size

    _validate_config(config=config, wsi_path=wsi_path)
    _prepare_output_dir(
        output_dir=output_dir,
        root_dir=root_dir,
        overwrite=config.overwrite,
    )

    selected_dir = output_dir / "selected"
    candidate_metadata_path = output_dir / "candidate_metadata.csv"
    selected_metadata_path = output_dir / "selected_metadata.csv"
    summary_path = output_dir / "selection_summary.json"
    method_config_path = output_dir / "method_config.json"
    preview_path = output_dir / "patch_selection_preview.png"

    write_json_manifest(
        _method_config(config, min_distance_level0=min_distance_level0),
        method_config_path,
    )

    openslide_module = _import_openslide()
    slide = openslide_module.OpenSlide(str(wsi_path))
    try:
        slide_metadata = _base_slide_metadata(slide)
        slide_dimensions = (
            int(slide_metadata["slide_width"]),
            int(slide_metadata["slide_height"]),
        )

        thumbnail = slide.get_thumbnail(
            (config.thumbnail_max_size, config.thumbnail_max_size)
        ).convert("RGB")
        candidates, num_candidates_generated = generate_tissue_candidates(
            thumbnail=thumbnail,
            slide_dimensions=slide_dimensions,
            patch_size=config.patch_size,
            stride=config.stride,
            min_tissue_ratio=config.min_tissue_ratio,
        )
        candidates_to_score = _select_candidates_to_score(
            candidates,
            seed=config.seed,
            max_candidates_to_score=config.max_candidates_to_score,
        )

        candidate_rows = [
            _candidate_pool_row(
                candidate,
                config=config,
                wsi_path=wsi_path,
                slide_metadata=slide_metadata,
            )
            for candidate in candidates
        ]
        candidate_rows_by_id = {
            str(row["candidate_id"]): row
            for row in candidate_rows
        }

        scored_records: list[dict[str, object]] = []
        for candidate in candidates_to_score:
            patch_image = slide.read_region(
                (candidate.x_level0, candidate.y_level0),
                0,
                (config.patch_size, config.patch_size),
            ).convert("RGB")
            features = compute_patch_features(
                rgb_patch=patch_image,
                feature_size=config.feature_size,
                nuclear_proxy=config.nuclear_proxy,
            )
            row = candidate_rows_by_id[candidate.candidate_id]
            row.update(
                {
                    "width": patch_image.width,
                    "height": patch_image.height,
                    "evaluated": True,
                    "scored": True,
                    "tissue_ratio": _format_float(features["tissue_ratio"]),
                    "nuclear_signal": _format_float(features["nuclear_signal"]),
                    "visual_entropy": _format_float(features["visual_entropy"]),
                    "blur_score": _format_float(features["blur_score"]),
                    "artifact_penalty": _format_float(features["artifact_penalty"]),
                }
            )
            scored_record = {
                "candidate_id": candidate.candidate_id,
                "x_level0": candidate.x_level0,
                "y_level0": candidate.y_level0,
                "patch_size": candidate.patch_size,
                "thumbnail_tissue_ratio": candidate.thumbnail_tissue_ratio,
                **features,
            }
            scored_records.append(scored_record)
            del patch_image

        apply_feature_scores(scored_records, weights=DEFAULT_SMART_WEIGHTS)
        x_norm = normalize_feature(record["x_level0"] for record in scored_records)
        y_norm = normalize_feature(record["y_level0"] for record in scored_records)
        thumbnail_norm = normalize_feature(
            record["thumbnail_tissue_ratio"] for record in scored_records
        )
        for index, record in enumerate(scored_records):
            record["x_norm"] = x_norm[index]
            record["y_norm"] = y_norm[index]
            record["thumbnail_tissue_ratio_norm"] = thumbnail_norm[index]

        quota_stats: dict[str, object] = {
            "regions_covered": None,
            "active_regions": None,
            "patches_per_region": {},
            "quota_grid": config.quota_grid if config.spatial_strategy == "quotas" else None,
            "quota_min_score_quantile": config.quota_min_score_quantile
            if config.spatial_strategy == "quotas"
            else None,
            "quota_fill_rate": None,
            "mean_feature_diversity_bonus_selected": None,
            "quota_score_threshold": None,
            "quota_eligible_candidates": None,
        }
        if config.spatial_strategy == "quotas":
            assign_spatial_regions(
                records=scored_records,
                slide_dimensions=slide_dimensions,
                quota_grid=config.quota_grid,
            )
            selected_records, quota_stats = select_with_spatial_quotas(
                records=scored_records,
                max_patches=config.max_patches,
                lambda_spatial=config.lambda_spatial,
                min_distance_level0=float(min_distance_level0),
                quota_min_score_quantile=config.quota_min_score_quantile,
                diversity_strategy=config.diversity_strategy,
                feature_diversity_weight=config.feature_diversity_weight,
                feature_names=FEATURE_DIVERSITY_FIELDS,
            )
        else:
            selected_records = greedy_select_with_spatial_penalty(
                records=scored_records,
                max_patches=config.max_patches,
                lambda_spatial=config.lambda_spatial,
                min_distance_level0=float(min_distance_level0),
                diversity_strategy=config.diversity_strategy,
                feature_diversity_weight=config.feature_diversity_weight,
                feature_names=FEATURE_DIVERSITY_FIELDS,
            )
            selected_region_ids = {
                str(record.get("region_id", ""))
                for record in selected_records
                if record.get("region_id")
            }
            if selected_records:
                bonuses = [
                    float(record.get("feature_diversity_bonus", 0.0))
                    for record in selected_records
                ]
                quota_stats["mean_feature_diversity_bonus_selected"] = (
                    sum(bonuses) / len(bonuses)
                )
            quota_stats["regions_covered"] = len(selected_region_ids) if selected_region_ids else None

        for record in scored_records:
            row = candidate_rows_by_id[str(record["candidate_id"])]
            row.update(
                {
                    "spatial_penalty": _format_float(record["spatial_penalty"]),
                    "feature_diversity_bonus": _format_float(
                        record.get("feature_diversity_bonus", 0.0)
                    ),
                    "score_raw": _format_float(record["score_raw"]),
                    "score_final": _format_float(record["score_final"]),
                    "region_id": record.get("region_id", ""),
                    "region_row": record.get("region_row", ""),
                    "region_col": record.get("region_col", ""),
                    "quota_grid": record.get("quota_grid", row.get("quota_grid", "")),
                    "selected": bool(record["selected"]),
                    "rank": record["rank"],
                }
            )

        selected_rows: list[dict[str, object]] = []
        for record in sorted(selected_records, key=lambda item: int(item["rank"])):
            row = candidate_rows_by_id[str(record["candidate_id"])]
            patch_id = f"patch_{len(selected_rows):04d}_x{row['x_level0']}_y{row['y_level0']}"
            filename = f"{patch_id}.png"
            patch_image = slide.read_region(
                (int(row["x_level0"]), int(row["y_level0"])),
                0,
                (config.patch_size, config.patch_size),
            ).convert("RGB")
            patch_image.save(selected_dir / filename)
            row["filename"] = filename
            selected_rows.append(_selected_row(row))
            del patch_image

        write_csv_manifest(
            rows=candidate_rows,
            output_path=candidate_metadata_path,
            fieldnames=CANDIDATE_METADATA_FIELDS,
        )
        write_csv_manifest(
            rows=selected_rows,
            output_path=selected_metadata_path,
            fieldnames=SELECTED_METADATA_FIELDS,
        )
        preview_rows = [
            row
            for row in candidate_rows
            if row["scored"] in (True, "True", "true", "1")
        ]
        save_wsi_patch_selection_preview(
            thumbnail=thumbnail,
            candidate_rows=preview_rows,
            slide_dimensions=slide_dimensions,
            output_path=preview_path,
        )

        summary: dict[str, Any] = {
            "selector": config.selector,
            "wsi_path": str(wsi_path),
            "output_dir": str(output_dir),
            "patch_size": config.patch_size,
            "stride": config.stride,
            "max_patches": config.max_patches,
            "min_tissue_ratio": config.min_tissue_ratio,
            "seed": config.seed,
            "thumbnail_max_size": config.thumbnail_max_size,
            "feature_size": config.feature_size,
            "max_candidates_to_score": config.max_candidates_to_score,
            "lambda_spatial": config.lambda_spatial,
            "min_distance_level0": min_distance_level0,
            **slide_metadata,
            "num_candidates_generated": num_candidates_generated,
            "num_thumbnail_candidates_passing_mask": len(candidates),
            "num_candidate_rows_written": len(candidate_rows),
            "num_candidates_scored": len(scored_records),
            "num_candidates_evaluated": len(scored_records),
            "num_selected": len(selected_rows),
            "mean_tissue_ratio_selected": _mean_selected(selected_rows, "tissue_ratio"),
            "mean_nuclear_signal_selected": _mean_selected(selected_rows, "nuclear_signal"),
            "mean_visual_entropy_selected": _mean_selected(selected_rows, "visual_entropy"),
            "mean_blur_score_selected": _mean_selected(selected_rows, "blur_score"),
            "mean_artifact_penalty_selected": _mean_selected(selected_rows, "artifact_penalty"),
            "mean_feature_diversity_bonus_selected": _mean_selected(
                selected_rows,
                "feature_diversity_bonus",
            ),
            "mean_score_raw_selected": _mean_selected(selected_rows, "score_raw"),
            "mean_score_final_selected": _mean_selected(selected_rows, "score_final"),
            "nuclear_proxy": config.nuclear_proxy,
            "spatial_strategy": config.spatial_strategy,
            "quota_grid": config.quota_grid if config.spatial_strategy == "quotas" else None,
            "quota_min_score_quantile": config.quota_min_score_quantile
            if config.spatial_strategy == "quotas"
            else None,
            "regions_covered": quota_stats.get("regions_covered"),
            "active_regions": quota_stats.get("active_regions"),
            "patches_per_region": quota_stats.get("patches_per_region"),
            "quota_fill_rate": quota_stats.get("quota_fill_rate"),
            "quota_score_threshold": quota_stats.get("score_threshold"),
            "quota_eligible_candidates": quota_stats.get("eligible_candidates"),
            "diversity_strategy": config.diversity_strategy,
            "feature_diversity_weight": config.feature_diversity_weight,
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "candidate_metadata_csv": str(candidate_metadata_path),
            "selected_metadata_csv": str(selected_metadata_path),
            "method_config_json": str(method_config_path),
            "preview_image": str(preview_path),
            "selected_dir": str(selected_dir),
            "candidate_pool": CANDIDATE_POOL,
            "candidate_metadata_semantics": CANDIDATE_METADATA_SEMANTICS,
            "preview_shows": PREVIEW_SHOWS,
            "candidate_ordering": CANDIDATE_ORDERING,
            "tissue_mask_method": TISSUE_MASK_METHOD,
            "nuclear_signal_method": nuclear_signal_method_for_proxy(config.nuclear_proxy),
            "visual_entropy_method": VISUAL_ENTROPY_METHOD,
            "blur_score_method": BLUR_SCORE_METHOD,
            "artifact_penalty_method": ARTIFACT_PENALTY_METHOD,
            "clinical_warning": CLINICAL_WARNING,
        }
        write_json_manifest(summary, summary_path)
        return summary
    finally:
        slide.close()
