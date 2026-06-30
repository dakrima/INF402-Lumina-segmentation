"""Baseline reproducible de selección de patches con TIAToolbox y máscara Otsu."""

from __future__ import annotations

import csv
import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.preprocessing.wsi_patch_extraction import (
    SUPPORTED_WSI_EXTENSIONS,
    clear_output_dir_safely,
    _slide_property,
)
from src.selection.candidate_generation import PatchCandidate
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
TIATOOLBOX_EXTRACTION_BACKEND = "tiatoolbox"
TIATOOLBOX_EXTRACTOR_NAME = "SlidingWindowPatchExtractor"
TIATOOLBOX_INPUT_MASK = "otsu"
TIATOOLBOX_TISSUE_MASK_METHOD = "tiatoolbox_otsu"
TIATOOLBOX_CANDIDATE_ORDERING = "tiatoolbox_otsu_candidates_seeded_shuffle"
TIATOOLBOX_CANDIDATE_POOL = "tiatoolbox_otsu_sliding_window_min_mask_ratio"
TIATOOLBOX_CANDIDATE_METADATA_SEMANTICS = "all_tiatoolbox_otsu_candidates"
PREVIEW_SHOWS = "selected_candidates"
@dataclass(frozen=True)
class TiatoolboxCandidate:
    """Coordenada candidata en nivel 0 generada por TIAToolbox."""

    candidate_id: str
    grid_index: int
    x_level0: int
    y_level0: int
    patch_size: int
    tiatoolbox_index: int


@dataclass
class SharedCandidatePool:
    """Pool TIAToolbox/Otsu compartido exactamente por ambos métodos."""

    case_id: str
    wsi_path: Path
    candidates: tuple[TiatoolboxCandidate, ...]
    extractor: object | None
    tiatoolbox_version: str
    slide_metadata: dict[str, Any]
    candidate_pool_hash: str
    candidate_generation_seconds: float

    def release_extractor(self) -> None:
        """Libera el handle de la WSI después de exportar el baseline."""
        if self.extractor is None:
            return
        wsi = getattr(self.extractor, "wsi", None)
        close = getattr(wsi, "close", None)
        if callable(close):
            close()
        self.extractor = None


@dataclass(frozen=True)
class BaselineSelectionConfig:
    """Configuración inmutable del baseline TIAToolbox."""

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
    """Resuelve una salida relativa respecto de la raíz del proyecto."""
    if output_dir.is_absolute():
        return output_dir.expanduser().resolve()
    return (root_dir / output_dir).resolve()


def _has_user_outputs(output_dir: Path) -> bool:
    """Indica si una carpeta contiene archivos distintos de `.gitkeep`."""
    if not output_dir.exists():
        return False
    return any(child.name != ".gitkeep" for child in output_dir.iterdir())


def _prepare_output_dir(output_dir: Path, root_dir: Path, overwrite: bool) -> None:
    """Valida, limpia cuando corresponde y crea la carpeta de salida del método."""
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
    """Valida selector, rangos numéricos, extensión y existencia de la WSI."""
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
    """Extrae dimensiones, niveles, objetivo y MPP desde una WSI OpenSlide."""
    slide_width, slide_height = slide.dimensions
    return {
        "slide_width": slide_width,
        "slide_height": slide_height,
        "level_count": slide.level_count,
        "objective_power": _slide_property(slide, "openslide.objective-power"),
        "mpp_x": _slide_property(slide, "openslide.mpp-x"),
        "mpp_y": _slide_property(slide, "openslide.mpp-y"),
    }


def _configure_tiatoolbox_runtime() -> None:
    """Configura caches escribibles antes de importar TIAToolbox."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")


def _import_tiatoolbox_extractor() -> tuple[type[Any], str]:
    """Importa de forma diferida el extractor de ventana deslizante."""
    _configure_tiatoolbox_runtime()
    try:
        import tiatoolbox
        from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic
        raise RuntimeError(
            "Missing or unusable dependency: TIAToolbox. Activate the "
            "inf402-lumina-seg environment before running baseline_tiatoolbox."
        ) from exc
    return SlidingWindowPatchExtractor, str(getattr(tiatoolbox, "__version__", "unknown"))


def _as_optional_number(value: object) -> int | float | str | None:
    """Convierte escalares NumPy a tipos serializables sin perder valores opcionales."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _slide_metadata_from_extractor(extractor: object) -> dict[str, Any]:
    """Lee la metadata expuesta por el lector WSI de TIAToolbox."""
    wsi = getattr(extractor, "wsi", None)
    info = getattr(wsi, "info", None)
    slide_dimensions = getattr(info, "slide_dimensions", None)
    level_dimensions = getattr(info, "level_dimensions", None)
    mpp = getattr(info, "mpp", None)

    if slide_dimensions is None and level_dimensions:
        slide_dimensions = level_dimensions[0]
    slide_width = int(slide_dimensions[0]) if slide_dimensions else None
    slide_height = int(slide_dimensions[1]) if slide_dimensions else None

    mpp_x = None
    mpp_y = None
    if mpp is not None and len(mpp) >= 2:
        mpp_x = _as_optional_number(mpp[0])
        mpp_y = _as_optional_number(mpp[1])

    level_count = getattr(info, "level_count", None)
    if level_count is None and level_dimensions is not None:
        level_count = len(level_dimensions)

    return {
        "slide_width": slide_width,
        "slide_height": slide_height,
        "level_count": _as_optional_number(level_count),
        "objective_power": _as_optional_number(getattr(info, "objective_power", None)),
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
    }


def _build_tiatoolbox_extractor(
    *,
    wsi_path: Path,
    config: BaselineSelectionConfig,
) -> tuple[object, str]:
    """Construye `SlidingWindowPatchExtractor` con máscara Otsu en nivel 0."""
    SlidingWindowPatchExtractor, tiatoolbox_version = _import_tiatoolbox_extractor()
    extractor = SlidingWindowPatchExtractor(
        input_img=str(wsi_path),
        patch_size=(config.patch_size, config.patch_size),
        stride=(config.stride, config.stride),
        input_mask=TIATOOLBOX_INPUT_MASK,
        min_mask_ratio=config.min_tissue_ratio,
        resolution=0,
        units="level",
    )
    return extractor, tiatoolbox_version


def _candidates_from_extractor(extractor: object, patch_size: int) -> list[TiatoolboxCandidate]:
    """
    Convierte `locations_df` en candidatos trazables de nivel 0.

    Conserva el índice TIAToolbox para recuperar exactamente cada patch posteriormente.
    """
    locations_df = getattr(extractor, "locations_df", None)
    if locations_df is None:
        raise RuntimeError("TIAToolbox extractor did not expose locations_df.")
    missing_columns = {"x", "y"} - set(locations_df.columns)
    if missing_columns:
        raise RuntimeError(
            "TIAToolbox locations_df is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    candidates: list[TiatoolboxCandidate] = []
    for row_position, (tiatoolbox_index, row) in enumerate(locations_df.iterrows()):
        tiatoolbox_index = int(tiatoolbox_index)
        x_level0 = int(row["x"])
        y_level0 = int(row["y"])
        candidates.append(
            TiatoolboxCandidate(
                candidate_id=f"candidate_{tiatoolbox_index:06d}_x{x_level0}_y{y_level0}",
                grid_index=int(row_position),
                x_level0=x_level0,
                y_level0=y_level0,
                patch_size=patch_size,
                tiatoolbox_index=tiatoolbox_index,
            )
        )
    return candidates


def canonical_candidate_rows(
    case_id: str,
    candidates: tuple[TiatoolboxCandidate, ...] | list[TiatoolboxCandidate],
) -> list[tuple[str, int, int, int, int]]:
    """Retorna filas canónicas para igualdad exacta y hash del pool."""
    return sorted(
        (
            case_id,
            int(candidate.x_level0),
            int(candidate.y_level0),
            0,
            int(candidate.patch_size),
        )
        for candidate in candidates
    )


def candidate_pool_hash(
    case_id: str,
    candidates: tuple[TiatoolboxCandidate, ...] | list[TiatoolboxCandidate],
) -> str:
    """Calcula SHA-256 sobre el caso y la geometría exacta del pool."""
    digest = hashlib.sha256()
    for row in canonical_candidate_rows(case_id, candidates):
        digest.update((",".join(str(value) for value in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def generate_shared_candidate_pool(
    config: BaselineSelectionConfig,
    *,
    case_id: str,
) -> SharedCandidatePool:
    """
    ***
    * config: Parámetros del baseline que definen patch, stride y filtro Otsu.
    * case_id: Identificador estable de la WSI.
    ***
    Genera una única vez el pool común con `SlidingWindowPatchExtractor`.
    Retorna candidatos, metadata, hash, extractor y tiempo de generación.
    """
    started = time.perf_counter()
    wsi_path = config.wsi_path.expanduser().resolve()
    extractor, tiatoolbox_version = _build_tiatoolbox_extractor(
        wsi_path=wsi_path,
        config=config,
    )
    slide_metadata = _slide_metadata_from_extractor(extractor)
    candidates = tuple(
        _candidates_from_extractor(
            extractor=extractor,
            patch_size=config.patch_size,
        )
    )
    return SharedCandidatePool(
        case_id=case_id,
        wsi_path=wsi_path,
        candidates=candidates,
        extractor=extractor,
        tiatoolbox_version=tiatoolbox_version,
        slide_metadata=slide_metadata,
        candidate_pool_hash=candidate_pool_hash(case_id, candidates),
        candidate_generation_seconds=round(time.perf_counter() - started, 6),
    )


def write_shared_candidate_manifest(
    pool: SharedCandidatePool,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Guarda el pool común y su metadata de reproducibilidad en CSV y JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shared_candidates.csv"
    json_path = output_dir / "shared_candidates.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "case_id",
            "generation_order",
            "candidate_id",
            "tiatoolbox_index",
            "x_level0",
            "y_level0",
            "level",
            "patch_size",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for order, candidate in enumerate(pool.candidates):
            writer.writerow(
                {
                    "case_id": pool.case_id,
                    "generation_order": order,
                    "candidate_id": candidate.candidate_id,
                    "tiatoolbox_index": candidate.tiatoolbox_index,
                    "x_level0": candidate.x_level0,
                    "y_level0": candidate.y_level0,
                    "level": 0,
                    "patch_size": candidate.patch_size,
                }
            )
    write_json_manifest(
        {
            "case_id": pool.case_id,
            "wsi_path": str(pool.wsi_path),
            "candidate_pool_hash": pool.candidate_pool_hash,
            "candidate_pool_hash_algorithm": "sha256",
            "candidate_pool_hash_fields": [
                "case_id",
                "x_level0",
                "y_level0",
                "level",
                "patch_size",
            ],
            "candidate_pool_count": len(pool.candidates),
            "candidate_generation_seconds": pool.candidate_generation_seconds,
            "level": 0,
            "patch_size": pool.candidates[0].patch_size if pool.candidates else None,
        },
        json_path,
    )
    return csv_path, json_path


def _patch_image_from_extractor(extractor: object, tiatoolbox_index: int) -> Image.Image:
    """Recupera un patch por índice TIAToolbox y lo normaliza a imagen RGB."""
    patch = extractor[tiatoolbox_index]
    if isinstance(patch, Image.Image):
        return patch.convert("RGB")
    patch_array = np.asarray(patch)
    if patch_array.ndim != 3 or patch_array.shape[2] < 3:
        raise RuntimeError(
            f"Unexpected TIAToolbox patch shape at index {tiatoolbox_index}: "
            f"{patch_array.shape}."
        )
    return Image.fromarray(patch_array[..., :3].astype(np.uint8), mode="RGB")


def _thumbnail_from_extractor(extractor: object, max_size: int) -> Image.Image | None:
    """Obtiene un thumbnail RGB acotado o retorna `None` si no está disponible."""
    wsi = getattr(extractor, "wsi", None)
    if wsi is None:
        return None
    try:
        thumbnail = wsi.slide_thumbnail()
    except Exception:
        return None
    thumbnail_array = np.asarray(thumbnail)
    if isinstance(thumbnail, Image.Image):
        image = thumbnail.convert("RGB")
    elif thumbnail_array.ndim == 3 and thumbnail_array.shape[2] >= 3:
        image = Image.fromarray(thumbnail_array[..., :3].astype(np.uint8), mode="RGB")
    else:
        return None
    image.thumbnail((max_size, max_size))
    return image


def _candidate_pool_row(
    candidate: PatchCandidate | TiatoolboxCandidate,
    *,
    config: BaselineSelectionConfig,
    wsi_path: Path,
    slide_metadata: dict[str, Any],
) -> dict[str, object]:
    """Construye la fila inicial de metadata para un candidato del pool común."""
    thumbnail_tissue_ratio = getattr(candidate, "thumbnail_tissue_ratio", "")
    if isinstance(thumbnail_tissue_ratio, float):
        thumbnail_tissue_ratio = f"{thumbnail_tissue_ratio:.6f}"
    row: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "grid_index": candidate.grid_index,
        "x_level0": candidate.x_level0,
        "y_level0": candidate.y_level0,
        "patch_size": candidate.patch_size,
        "stride": config.stride,
        "tiatoolbox_index": getattr(candidate, "tiatoolbox_index", ""),
        "min_mask_ratio": config.min_tissue_ratio,
        "width": "",
        "height": "",
        "thumbnail_tissue_ratio": thumbnail_tissue_ratio,
        "evaluated": False,
        "tissue_ratio": "",
        "selected": False,
        "rank": "",
        "filename": "",
        "selector": config.selector,
        "selection_method": config.selector,
        "seed": config.seed,
        "source_wsi_path": str(wsi_path),
        "tissue_mask_method": TIATOOLBOX_TISSUE_MASK_METHOD,
        **slide_metadata,
    }
    return row


def _selected_row(
    candidate_row: dict[str, object],
    *,
    patch_id: str,
    filename: str,
) -> dict[str, object]:
    """Construye la fila persistida de un patch seleccionado por el baseline."""
    return {
        "patch_id": patch_id,
        "filename": filename,
        "selected": True,
        "rank": candidate_row["rank"],
        "x_level0": candidate_row["x_level0"],
        "y_level0": candidate_row["y_level0"],
        "patch_size": candidate_row["patch_size"],
        "stride": candidate_row.get("stride", ""),
        "tiatoolbox_index": candidate_row.get("tiatoolbox_index", ""),
        "min_mask_ratio": candidate_row.get("min_mask_ratio", ""),
        "width": candidate_row["width"],
        "height": candidate_row["height"],
        "thumbnail_tissue_ratio": candidate_row["thumbnail_tissue_ratio"],
        "tissue_ratio": candidate_row["tissue_ratio"],
        "selector": candidate_row.get("selector", ""),
        "source_wsi_path": candidate_row["source_wsi_path"],
        "slide_width": candidate_row["slide_width"],
        "slide_height": candidate_row["slide_height"],
        "objective_power": candidate_row["objective_power"],
        "mpp_x": candidate_row["mpp_x"],
        "mpp_y": candidate_row["mpp_y"],
        "level_count": candidate_row["level_count"],
        "selection_method": candidate_row["selection_method"],
        "seed": candidate_row["seed"],
        "tissue_mask_method": candidate_row.get("tissue_mask_method", ""),
    }


def _method_config(
    config: BaselineSelectionConfig,
    *,
    tiatoolbox_version: str | None = None,
) -> dict[str, object]:
    """Registra configuración, versión TIAToolbox y definición del pool baseline."""
    return {
        "selector": config.selector,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "min_mask_ratio": config.min_tissue_ratio,
        "seed": config.seed,
        "thumbnail_max_size": config.thumbnail_max_size,
        "candidate_ordering": TIATOOLBOX_CANDIDATE_ORDERING,
        "candidate_pool": TIATOOLBOX_CANDIDATE_POOL,
        "candidate_metadata_semantics": TIATOOLBOX_CANDIDATE_METADATA_SEMANTICS,
        "extraction_backend": TIATOOLBOX_EXTRACTION_BACKEND,
        "tiatoolbox_version": tiatoolbox_version,
        "tiatoolbox_extractor": TIATOOLBOX_EXTRACTOR_NAME,
        "input_mask": TIATOOLBOX_INPUT_MASK,
        "tissue_mask_method": TIATOOLBOX_TISSUE_MASK_METHOD,
        "created_at": utc_now_iso(),
    }


def run_baseline_selection(
    config: BaselineSelectionConfig,
    *,
    shared_pool: SharedCandidatePool | None = None,
) -> dict[str, Any]:
    """
    ***
    * config: Parámetros fijos del baseline.
    * shared_pool: Pool común ya generado para la misma WSI.
    ***
    Selecciona de forma reproducible con semilla 42, guarda los 16 patches y sus
    manifiestos, configuración y preview. Retorna el resumen de la selección.
    """
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

    if shared_pool is None:
        generated_pool = generate_shared_candidate_pool(config, case_id=wsi_path.stem)
        extractor = generated_pool.extractor
        tiatoolbox_version = generated_pool.tiatoolbox_version
        slide_metadata = generated_pool.slide_metadata
        candidates = list(generated_pool.candidates)
        pool_hash = generated_pool.candidate_pool_hash
        candidate_generation_seconds = generated_pool.candidate_generation_seconds
    else:
        if shared_pool.wsi_path != wsi_path:
            raise ValueError("Shared candidate pool belongs to a different WSI.")
        extractor = shared_pool.extractor
        if extractor is None:
            raise RuntimeError("Shared TIAToolbox extractor was released before baseline export.")
        tiatoolbox_version = shared_pool.tiatoolbox_version
        slide_metadata = shared_pool.slide_metadata
        candidates = list(shared_pool.candidates)
        pool_hash = shared_pool.candidate_pool_hash
        candidate_generation_seconds = shared_pool.candidate_generation_seconds
    selection_started = time.perf_counter()
    method_config = _method_config(config, tiatoolbox_version=tiatoolbox_version)
    method_config.update(
        {
            "candidate_pool_hash": pool_hash,
            "candidate_pool_count": len(candidates),
            "candidate_generation_seconds": candidate_generation_seconds,
        }
    )
    write_json_manifest(
        method_config,
        method_config_path,
    )

    slide_dimensions = (
        int(slide_metadata["slide_width"]),
        int(slide_metadata["slide_height"]),
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
    num_candidates_evaluated = 0
    preview_warning: str | None = None

    for candidate in ordered_candidates:
        if len(selected_rows) >= config.max_patches:
            break

        patch_image = _patch_image_from_extractor(
            extractor=extractor,
            tiatoolbox_index=candidate.tiatoolbox_index,
        )
        rank = len(selected_rows) + 1
        num_candidates_evaluated += 1

        row = candidate_rows_by_id[candidate.candidate_id]
        patch_id = f"patch_{len(selected_rows):04d}_x{candidate.x_level0}_y{candidate.y_level0}"
        filename = f"{patch_id}.png"
        row.update(
            {
                "width": patch_image.width,
                "height": patch_image.height,
                "evaluated": True,
                "selected": True,
                "rank": rank,
                "filename": filename,
                "patch_id": patch_id,
                "patch_path": str(selected_dir / filename),
            }
        )
        patch_image.save(selected_dir / filename)
        selected_rows.append(
            _selected_row(
                row,
                patch_id=patch_id,
                filename=filename,
            )
        )

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

    thumbnail = _thumbnail_from_extractor(
        extractor=extractor,
        max_size=config.thumbnail_max_size,
    )
    if thumbnail is not None:
        save_wsi_patch_selection_preview(
            thumbnail=thumbnail,
            candidate_rows=[
                row for row in candidate_rows
                if row["selected"] in (True, "True", "true", "1")
            ],
            slide_dimensions=slide_dimensions,
            output_path=preview_path,
        )
    else:
        preview_warning = "Preview not generated because TIAToolbox thumbnail was unavailable."

    baseline_selection_seconds = round(time.perf_counter() - selection_started, 6)
    summary: dict[str, Any] = {
        "selector": config.selector,
        "wsi_path": str(wsi_path),
        "output_dir": str(output_dir),
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_patches": config.max_patches,
        "min_tissue_ratio": config.min_tissue_ratio,
        "min_mask_ratio": config.min_tissue_ratio,
        "seed": config.seed,
        **slide_metadata,
        "num_candidates_generated": len(candidates),
        "num_thumbnail_candidates_passing_mask": len(candidates),
        "num_candidate_rows_written": len(candidate_rows),
        "num_candidates_evaluated": num_candidates_evaluated,
        "num_candidates_passing_tissue_filter": len(selected_rows),
        "num_selected": len(selected_rows),
        "candidate_pool_definition": (
            "TIAToolbox sliding-window candidates passing Otsu input mask and min_mask_ratio."
        ),
        "mean_tissue_ratio_selected": None,
        "min_tissue_ratio_selected": None,
        "max_tissue_ratio_selected": None,
        "candidate_pool_hash": pool_hash,
        "candidate_pool_count": len(candidates),
        "candidate_generation_seconds": candidate_generation_seconds,
        "baseline_selection_seconds": baseline_selection_seconds,
        "runtime_seconds": (
            baseline_selection_seconds
            if shared_pool is not None
            else round(time.perf_counter() - start_time, 6)
        ),
        "candidate_metadata_csv": str(candidate_metadata_path),
        "selected_metadata_csv": str(selected_metadata_path),
        "method_config_json": str(method_config_path),
        "preview_image": str(preview_path) if thumbnail is not None else None,
        "selected_dir": str(selected_dir),
        "candidate_ordering": TIATOOLBOX_CANDIDATE_ORDERING,
        "candidate_pool": TIATOOLBOX_CANDIDATE_POOL,
        "candidate_metadata_semantics": TIATOOLBOX_CANDIDATE_METADATA_SEMANTICS,
        "extraction_backend": TIATOOLBOX_EXTRACTION_BACKEND,
        "tiatoolbox_version": tiatoolbox_version,
        "tiatoolbox_extractor": TIATOOLBOX_EXTRACTOR_NAME,
        "input_mask": TIATOOLBOX_INPUT_MASK,
        "preview_shows": PREVIEW_SHOWS,
        "preview_warning": preview_warning,
        "tissue_mask_method": TIATOOLBOX_TISSUE_MASK_METHOD,
    }
    write_json_manifest(summary, summary_path)
    return summary
