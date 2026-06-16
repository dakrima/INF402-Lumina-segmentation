"""Baseline TIAToolbox-style WSI patch selection.

This module implements only the reproducible baseline for INF402 Etapa 1:
grid candidates, thumbnail tissue filtering, patch-level tissue filtering,
bounded patch export, metadata, summary JSON, method config, and preview.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.wsi_patch_extraction import (
    SUPPORTED_WSI_EXTENSIONS,
    TISSUE_MASK_METHOD,
    clear_output_dir_safely,
    compute_simple_tissue_ratio,
    _import_openslide,
    _slide_property,
)
from src.selection.candidate_generation import PatchCandidate, generate_tissue_candidates
from src.selection.manifests import (
    CANDIDATE_METADATA_FIELDS,
    SELECTED_METADATA_FIELDS,
    utc_now_iso,
    write_csv_manifest,
    write_json_manifest,
)
from src.selection.previews import save_wsi_patch_selection_preview


BASELINE_SELECTOR_NAME = "baseline_tiatoolbox"
CANDIDATE_ORDERING = "thumbnail_filtered_seeded_shuffle"
CANDIDATE_POOL = "thumbnail_tissue_mask_filtered"
CANDIDATE_METADATA_SEMANTICS = "all_thumbnail_filtered_candidates"
PREVIEW_SHOWS = "evaluated_candidates"
CLINICAL_WARNING = (
    "Technical patch selection only. Not for diagnosis, not RCB, not clinical validation."
)


@dataclass(frozen=True)
class BaselineSelectionConfig:
    """Configuration for the baseline TIAToolbox-style selector."""

    wsi_path: Path
    output_dir: Path
    root_dir: Path
    selector: str = BASELINE_SELECTOR_NAME
    patch_size: int = 1024
    stride: int = 1024
    max_patches: int = 16
    min_tissue_ratio: float = 0.20
    seed: int = 42
    thumbnail_max_size: int = 2048
    overwrite: bool = False


def _resolve_output_dir(output_dir: Path, root_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir.expanduser().resolve()
    return (root_dir / output_dir).resolve()


def _has_user_outputs(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(child.name != ".gitkeep" for child in output_dir.iterdir())


def _prepare_output_dir(output_dir: Path, root_dir: Path, overwrite: bool) -> None:
    if _has_user_outputs(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite to regenerate this run."
        )
    if overwrite and output_dir.exists():
        clear_output_dir_safely(output_dir=output_dir, root_dir=root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected").mkdir(parents=True, exist_ok=True)


def _validate_config(config: BaselineSelectionConfig, wsi_path: Path) -> None:
    if config.selector != BASELINE_SELECTOR_NAME:
        raise NotImplementedError(
            f"Selector '{config.selector}' todavía no está implementado. "
            f"Esta etapa solo soporta {BASELINE_SELECTOR_NAME}."
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
    if wsi_path.suffix.lower() not in SUPPORTED_WSI_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_WSI_EXTENSIONS))
        raise ValueError(f"Unsupported WSI extension '{wsi_path.suffix}'. Use one of: {allowed}.")
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")


def _base_slide_metadata(slide: object) -> dict[str, Any]:
    slide_width, slide_height = slide.dimensions
    return {
        "slide_width": slide_width,
        "slide_height": slide_height,
        "level_count": slide.level_count,
        "objective_power": _slide_property(slide, "openslide.objective-power"),
        "mpp_x": _slide_property(slide, "openslide.mpp-x"),
        "mpp_y": _slide_property(slide, "openslide.mpp-y"),
    }


def _candidate_pool_row(
    candidate: PatchCandidate,
    *,
    config: BaselineSelectionConfig,
    wsi_path: Path,
    slide_metadata: dict[str, Any],
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "grid_index": candidate.grid_index,
        "x_level0": candidate.x_level0,
        "y_level0": candidate.y_level0,
        "patch_size": candidate.patch_size,
        "width": "",
        "height": "",
        "thumbnail_tissue_ratio": f"{candidate.thumbnail_tissue_ratio:.6f}",
        "evaluated": False,
        "tissue_ratio": "",
        "selected": False,
        "rank": "",
        "filename": "",
        "selection_method": config.selector,
        "seed": config.seed,
        "source_wsi_path": str(wsi_path),
        **slide_metadata,
    }
    return row


def _selected_row(
    candidate_row: dict[str, object],
    *,
    patch_id: str,
    filename: str,
) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "filename": filename,
        "selected": True,
        "rank": candidate_row["rank"],
        "x_level0": candidate_row["x_level0"],
        "y_level0": candidate_row["y_level0"],
        "patch_size": candidate_row["patch_size"],
        "width": candidate_row["width"],
        "height": candidate_row["height"],
        "thumbnail_tissue_ratio": candidate_row["thumbnail_tissue_ratio"],
        "tissue_ratio": candidate_row["tissue_ratio"],
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


def _method_config(config: BaselineSelectionConfig) -> dict[str, object]:
    return {
        "selector": config.selector,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "seed": config.seed,
        "thumbnail_max_size": config.thumbnail_max_size,
        "candidate_ordering": CANDIDATE_ORDERING,
        "candidate_pool": CANDIDATE_POOL,
        "candidate_metadata_semantics": CANDIDATE_METADATA_SEMANTICS,
        "tissue_mask_method": TISSUE_MASK_METHOD,
        "created_at": utc_now_iso(),
    }


def run_baseline_selection(config: BaselineSelectionConfig) -> dict[str, Any]:
    """Run the baseline selector and write all required outputs."""
    start_time = time.perf_counter()
    root_dir = config.root_dir.resolve()
    wsi_path = config.wsi_path.expanduser().resolve()
    output_dir = _resolve_output_dir(config.output_dir, root_dir=root_dir)

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

    write_json_manifest(_method_config(config), method_config_path)

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

        ordered_candidates = list(candidates)
        random.Random(config.seed).shuffle(ordered_candidates)

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
        selected_rows: list[dict[str, object]] = []
        selected_tissue_ratios: list[float] = []
        num_candidates_evaluated = 0
        num_candidates_passing_tissue_filter = 0

        for candidate in ordered_candidates:
            if len(selected_rows) >= config.max_patches:
                break

            patch_image = slide.read_region(
                (candidate.x_level0, candidate.y_level0),
                0,
                (config.patch_size, config.patch_size),
            ).convert("RGB")
            tissue_ratio = compute_simple_tissue_ratio(patch_image)
            selected = tissue_ratio >= config.min_tissue_ratio
            rank = len(selected_rows) + 1 if selected else None
            num_candidates_evaluated += 1

            row = candidate_rows_by_id[candidate.candidate_id]
            row.update(
                {
                    "width": patch_image.width,
                    "height": patch_image.height,
                    "evaluated": True,
                    "tissue_ratio": f"{tissue_ratio:.6f}",
                    "selected": selected,
                    "rank": rank if rank is not None else "",
                }
            )

            if not selected:
                continue

            num_candidates_passing_tissue_filter += 1
            patch_id = f"patch_{len(selected_rows):04d}_x{candidate.x_level0}_y{candidate.y_level0}"
            filename = f"{patch_id}.png"
            row["filename"] = filename
            patch_image.save(selected_dir / filename)
            selected_rows.append(
                _selected_row(
                    row,
                    patch_id=patch_id,
                    filename=filename,
                )
            )
            selected_tissue_ratios.append(tissue_ratio)

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
        save_wsi_patch_selection_preview(
            thumbnail=thumbnail,
            candidate_rows=[
                row for row in candidate_rows
                if row["evaluated"] in (True, "True", "true", "1")
            ],
            slide_dimensions=slide_dimensions,
            output_path=preview_path,
        )

        mean_tissue_ratio_selected = (
            sum(selected_tissue_ratios) / len(selected_tissue_ratios)
            if selected_tissue_ratios
            else None
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
            **slide_metadata,
            "num_candidates_generated": num_candidates_generated,
            "num_thumbnail_candidates_passing_mask": len(candidates),
            "num_candidate_rows_written": len(candidate_rows),
            "num_candidates_evaluated": num_candidates_evaluated,
            "num_candidates_passing_tissue_filter": num_candidates_passing_tissue_filter,
            "num_selected": len(selected_rows),
            "candidate_pool_definition": (
                "Grid candidates that passed thumbnail tissue mask using min_tissue_ratio."
            ),
            "mean_tissue_ratio_selected": mean_tissue_ratio_selected,
            "min_tissue_ratio_selected": min(selected_tissue_ratios)
            if selected_tissue_ratios
            else None,
            "max_tissue_ratio_selected": max(selected_tissue_ratios)
            if selected_tissue_ratios
            else None,
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "candidate_metadata_csv": str(candidate_metadata_path),
            "selected_metadata_csv": str(selected_metadata_path),
            "method_config_json": str(method_config_path),
            "preview_image": str(preview_path),
            "selected_dir": str(selected_dir),
            "candidate_ordering": CANDIDATE_ORDERING,
            "candidate_pool": CANDIDATE_POOL,
            "candidate_metadata_semantics": CANDIDATE_METADATA_SEMANTICS,
            "preview_shows": PREVIEW_SHOWS,
            "tissue_mask_method": TISSUE_MASK_METHOD,
            "clinical_warning": CLINICAL_WARNING,
        }
        write_json_manifest(summary, summary_path)
        return summary
    finally:
        slide.close()
