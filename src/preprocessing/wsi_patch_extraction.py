"""OpenSlide-based WSI patch extraction for reproducible smoke tests."""

from __future__ import annotations

import csv
import importlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.visualization.patch_preview import PatchBox, save_patch_selection_preview


CLINICAL_WARNING = (
    "Technical patch extraction only. Not for diagnosis, not RCB, not clinical validation."
)
SELECTION_METHOD = "thumbnail_tissue_mask_grid"
TISSUE_MASK_METHOD = "mean_lt_235_and_std_gt_8"
SUPPORTED_WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif"}


@dataclass(frozen=True)
class WsiPatchExtractionConfig:
    """Configuration for a bounded WSI patch extraction run."""

    wsi_path: Path
    output_dir: Path
    root_dir: Path
    patch_size: int = 1024
    max_patches: int = 8
    min_tissue_ratio: float = 0.2
    thumbnail_size: int = 2048
    clear_output: bool = False
    preview_image: bool = False
    save_rejected: bool = False
    seed: int = 42


@dataclass(frozen=True)
class WsiPatchCandidate:
    """Level-0 WSI patch coordinate plus thumbnail tissue score."""

    x_level0: int
    y_level0: int
    thumbnail_tissue_ratio: float


def _import_openslide() -> object:
    try:
        return importlib.import_module("openslide")
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic
        raise RuntimeError(
            "Missing dependency: openslide. Activate the inf402-lumina-seg Conda/Mamba "
            "environment before extracting WSI patches."
        ) from exc


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def clear_output_dir_safely(output_dir: Path, root_dir: Path) -> None:
    """Clear only a non-dangerous output directory inside the repository."""
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        resolved_root,
        resolved_root / "data",
        resolved_root / "outputs",
    }

    if not _is_relative_to(resolved_output, resolved_root):
        raise ValueError("--clear-output only supports output directories inside the repository.")
    if resolved_output in dangerous_paths:
        raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")

    resolved_output.mkdir(parents=True, exist_ok=True)
    for child in resolved_output.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def estimate_thumbnail_tissue_mask(rgb_image: Image.Image | np.ndarray) -> np.ndarray:
    """Estimate tissue on a thumbnail using a simple non-clinical RGB rule."""
    array = np.asarray(rgb_image.convert("RGB") if isinstance(rgb_image, Image.Image) else rgb_image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")

    rgb = array[..., :3].astype(np.float32)
    mean = np.mean(rgb, axis=-1)
    std = np.std(rgb, axis=-1)
    return (mean < 235) & (std > 8)


def compute_simple_tissue_ratio(rgb_image: Image.Image | np.ndarray) -> float:
    """Compute the approximate fraction of tissue using the thumbnail mask rule."""
    tissue_mask = estimate_thumbnail_tissue_mask(rgb_image)
    if tissue_mask.size == 0:
        return 0.0
    return float(np.mean(tissue_mask))


def _resolve_output_dir(output_dir: Path, root_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir.resolve()
    return (root_dir / output_dir).resolve()


def _slide_property(slide: object, property_name: str) -> str | None:
    properties = getattr(slide, "properties", {})
    value = properties.get(property_name)
    if value is None:
        return None
    return str(value)


def _base_summary(config: WsiPatchExtractionConfig, output_dir: Path) -> dict[str, Any]:
    return {
        "status": "failed",
        "wsi_path": str(config.wsi_path),
        "slide_width": None,
        "slide_height": None,
        "level_count": None,
        "level_dimensions": [],
        "objective_power": None,
        "mpp_x": None,
        "mpp_y": None,
        "patch_size": config.patch_size,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "thumbnail_size": config.thumbnail_size,
        "seed": config.seed,
        "num_grid_candidates": 0,
        "num_thumbnail_candidates": 0,
        "num_candidates_evaluated": 0,
        "num_selected": 0,
        "num_rejected": 0,
        "selection_method": SELECTION_METHOD,
        "tissue_mask_method": TISSUE_MASK_METHOD,
        "output_dir": str(output_dir),
        "selected_dir": str(output_dir / "selected"),
        "rejected_dir": str(output_dir / "rejected") if config.save_rejected else None,
        "metadata_csv": str(output_dir / "patches_metadata.csv"),
        "summary_json": str(output_dir / "summary.json"),
        "preview_image": str(output_dir / "patch_selection_preview.png")
        if config.preview_image
        else None,
        "clinical_warning": CLINICAL_WARNING,
        "error": None,
        "suggested_next_step": None,
    }


def _thumbnail_bbox_for_patch(
    candidate: tuple[int, int],
    patch_size: int,
    slide_dimensions: tuple[int, int],
    thumbnail_dimensions: tuple[int, int],
) -> tuple[int, int, int, int]:
    slide_width, slide_height = slide_dimensions
    thumb_width, thumb_height = thumbnail_dimensions
    x_level0, y_level0 = candidate
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height
    x0 = int(np.floor(x_level0 * scale_x))
    y0 = int(np.floor(y_level0 * scale_y))
    x1 = int(np.ceil((x_level0 + patch_size) * scale_x))
    y1 = int(np.ceil((y_level0 + patch_size) * scale_y))
    x0 = max(0, min(thumb_width - 1, x0))
    y0 = max(0, min(thumb_height - 1, y0))
    x1 = max(x0 + 1, min(thumb_width, x1))
    y1 = max(y0 + 1, min(thumb_height, y1))
    return x0, y0, x1, y1


def _generate_grid_coordinates(
    slide_width: int,
    slide_height: int,
    patch_size: int,
) -> list[tuple[int, int]]:
    if slide_width < patch_size or slide_height < patch_size:
        return []
    x_positions = range(0, slide_width - patch_size + 1, patch_size)
    y_positions = range(0, slide_height - patch_size + 1, patch_size)
    return [(x, y) for y in y_positions for x in x_positions]


def _select_thumbnail_candidates(
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    patch_size: int,
    min_tissue_ratio: float,
) -> tuple[list[WsiPatchCandidate], int]:
    slide_width, slide_height = slide_dimensions
    thumbnail_dimensions = (tissue_mask.shape[1], tissue_mask.shape[0])
    grid_coordinates = _generate_grid_coordinates(
        slide_width=slide_width,
        slide_height=slide_height,
        patch_size=patch_size,
    )
    candidates: list[WsiPatchCandidate] = []

    for x_level0, y_level0 in grid_coordinates:
        x0, y0, x1, y1 = _thumbnail_bbox_for_patch(
            candidate=(x_level0, y_level0),
            patch_size=patch_size,
            slide_dimensions=slide_dimensions,
            thumbnail_dimensions=thumbnail_dimensions,
        )
        thumbnail_patch_mask = tissue_mask[y0:y1, x0:x1]
        thumbnail_tissue_ratio = (
            float(np.mean(thumbnail_patch_mask)) if thumbnail_patch_mask.size else 0.0
        )
        if thumbnail_tissue_ratio >= min_tissue_ratio:
            candidates.append(
                WsiPatchCandidate(
                    x_level0=x_level0,
                    y_level0=y_level0,
                    thumbnail_tissue_ratio=thumbnail_tissue_ratio,
                )
            )

    return candidates, len(grid_coordinates)


def _preview_boxes_from_rows(
    rows: list[dict[str, object]],
    slide_dimensions: tuple[int, int],
    thumbnail_dimensions: tuple[int, int],
) -> list[PatchBox]:
    slide_width, slide_height = slide_dimensions
    thumb_width, thumb_height = thumbnail_dimensions
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height
    boxes: list[PatchBox] = []

    for row in rows:
        x = int(row["x_level0"])
        y = int(row["y_level0"])
        patch_size = int(row["patch_size"])
        selected = bool(row["selected"])
        x0 = int(round(x * scale_x))
        y0 = int(round(y * scale_y))
        width = max(1, int(round(patch_size * scale_x)))
        height = max(1, int(round(patch_size * scale_y)))
        boxes.append(PatchBox(x=x0, y=y0, width=width, height=height, selected=selected))

    return boxes


def _write_metadata_csv(rows: list[dict[str, object]], metadata_path: Path) -> None:
    fieldnames = [
        "patch_id",
        "filename",
        "selected",
        "saved",
        "x_level0",
        "y_level0",
        "patch_size",
        "width",
        "height",
        "tissue_ratio",
        "thumbnail_tissue_ratio",
        "source_wsi_path",
        "slide_width",
        "slide_height",
        "objective_power",
        "mpp_x",
        "mpp_y",
        "level_count",
        "selection_method",
    ]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_wsi_patches(config: WsiPatchExtractionConfig) -> tuple[dict[str, Any], Path]:
    """Extract a bounded set of level-0 WSI patches and write metadata."""
    wsi_path = config.wsi_path.expanduser().resolve()
    root_dir = config.root_dir.resolve()
    output_dir = _resolve_output_dir(config.output_dir, root_dir=root_dir)
    summary = _base_summary(config=config, output_dir=output_dir)
    summary["wsi_path"] = str(wsi_path)
    summary_path = output_dir / "summary.json"

    try:
        if config.patch_size <= 0:
            raise ValueError("--patch-size must be positive.")
        if config.max_patches <= 0:
            raise ValueError("--max-patches must be positive.")
        if not 0 <= config.min_tissue_ratio <= 1:
            raise ValueError("--min-tissue-ratio must be between 0 and 1.")
        if config.thumbnail_size <= 0:
            raise ValueError("--thumbnail-size must be positive.")
        if wsi_path.suffix.lower() not in SUPPORTED_WSI_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_WSI_EXTENSIONS))
            raise ValueError(f"Unsupported WSI extension '{wsi_path.suffix}'. Use one of: {allowed}.")
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI path does not exist: {wsi_path}")

        if config.clear_output:
            clear_output_dir_safely(output_dir=output_dir, root_dir=root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_dir = output_dir / "selected"
        rejected_dir = output_dir / "rejected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        if config.save_rejected:
            rejected_dir.mkdir(parents=True, exist_ok=True)

        openslide_module = _import_openslide()
        slide = openslide_module.OpenSlide(str(wsi_path))
        try:
            slide_width, slide_height = slide.dimensions
            level_dimensions = [list(dimensions) for dimensions in slide.level_dimensions]
            objective_power = _slide_property(slide, "openslide.objective-power")
            mpp_x = _slide_property(slide, "openslide.mpp-x")
            mpp_y = _slide_property(slide, "openslide.mpp-y")

            summary.update(
                {
                    "slide_width": slide_width,
                    "slide_height": slide_height,
                    "level_count": slide.level_count,
                    "level_dimensions": level_dimensions,
                    "objective_power": objective_power,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                }
            )

            thumbnail = slide.get_thumbnail((config.thumbnail_size, config.thumbnail_size)).convert("RGB")
            tissue_mask = estimate_thumbnail_tissue_mask(thumbnail)
            candidates, num_grid_candidates = _select_thumbnail_candidates(
                tissue_mask=tissue_mask,
                slide_dimensions=(slide_width, slide_height),
                patch_size=config.patch_size,
                min_tissue_ratio=config.min_tissue_ratio,
            )
            rng = random.Random(config.seed)
            rng.shuffle(candidates)

            rows: list[dict[str, object]] = []
            selected_count = 0
            rejected_count = 0
            evaluated_count = 0

            for candidate in candidates:
                if selected_count >= config.max_patches:
                    break

                patch_image = slide.read_region(
                    (candidate.x_level0, candidate.y_level0),
                    0,
                    (config.patch_size, config.patch_size),
                ).convert("RGB")
                tissue_ratio = compute_simple_tissue_ratio(patch_image)
                selected = tissue_ratio >= config.min_tissue_ratio
                evaluated_count += 1
                saved = False
                filename = ""

                if selected:
                    patch_id = (
                        f"patch_{selected_count:04d}_x{candidate.x_level0}_"
                        f"y{candidate.y_level0}"
                    )
                    filename = f"{patch_id}.png"
                    patch_image.save(selected_dir / filename)
                    selected_count += 1
                    saved = True
                else:
                    patch_id = (
                        f"rejected_{rejected_count:04d}_x{candidate.x_level0}_"
                        f"y{candidate.y_level0}"
                    )
                    rejected_count += 1
                    if config.save_rejected:
                        filename = f"{patch_id}.png"
                        patch_image.save(rejected_dir / filename)
                        saved = True

                rows.append(
                    {
                        "patch_id": patch_id,
                        "filename": filename,
                        "selected": selected,
                        "saved": saved,
                        "x_level0": candidate.x_level0,
                        "y_level0": candidate.y_level0,
                        "patch_size": config.patch_size,
                        "width": patch_image.width,
                        "height": patch_image.height,
                        "tissue_ratio": f"{tissue_ratio:.6f}",
                        "thumbnail_tissue_ratio": f"{candidate.thumbnail_tissue_ratio:.6f}",
                        "source_wsi_path": str(wsi_path),
                        "slide_width": slide_width,
                        "slide_height": slide_height,
                        "objective_power": objective_power,
                        "mpp_x": mpp_x,
                        "mpp_y": mpp_y,
                        "level_count": slide.level_count,
                        "selection_method": SELECTION_METHOD,
                    }
                )

            metadata_path = output_dir / "patches_metadata.csv"
            _write_metadata_csv(rows=rows, metadata_path=metadata_path)

            preview_path = output_dir / "patch_selection_preview.png"
            if config.preview_image:
                preview_boxes = _preview_boxes_from_rows(
                    rows=rows,
                    slide_dimensions=(slide_width, slide_height),
                    thumbnail_dimensions=thumbnail.size,
                )
                save_patch_selection_preview(
                    rgb_image=thumbnail,
                    patches=preview_boxes,
                    output_path=preview_path,
                )

            summary.update(
                {
                    "status": "completed",
                    "num_grid_candidates": num_grid_candidates,
                    "num_thumbnail_candidates": len(candidates),
                    "num_candidates_evaluated": evaluated_count,
                    "num_selected": selected_count,
                    "num_rejected": rejected_count,
                    "metadata_csv": str(metadata_path),
                    "preview_image": str(preview_path) if config.preview_image else None,
                    "error": None,
                    "suggested_next_step": (
                        "Run scripts/04_run_inference.py on one selected patch with "
                        "--input-mode patch for the next technical smoke test."
                    ),
                }
            )
        finally:
            slide.close()
    except Exception as exc:  # noqa: BLE001 - write diagnostic summary
        output_dir.mkdir(parents=True, exist_ok=True)
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["suggested_next_step"] = (
            "Activate the inf402-lumina-seg environment, verify OpenSlide can read the "
            "WSI, and rerun with a bounded --max-patches value."
        )

    write_summary_json(summary=summary, summary_path=summary_path)
    return summary, summary_path
