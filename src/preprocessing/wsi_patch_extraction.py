"""Funciones OpenSlide y de tejido compartidas por el experimento final."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif"}


def _import_openslide() -> object:
    """Importa OpenSlide de forma diferida y contextualiza errores de dependencia."""
    try:
        return importlib.import_module("openslide")
    except Exception as exc:  # noqa: BLE001 - diagnóstico de dependencia nativa
        raise RuntimeError(
            "No se pudo importar OpenSlide. Active el entorno inf402-lumina-seg."
        ) from exc


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Indica si una ruta se encuentra dentro de otra."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def clear_output_dir_safely(output_dir: Path, root_dir: Path) -> None:
    """
    ***
    * output_dir: Carpeta de una corrida que será regenerada.
    * root_dir: Raíz del repositorio usada como límite de seguridad.
    ***
    Vacía solamente una subcarpeta no crítica dentro del repositorio. Rechaza la raíz,
    `data`, `models` y `results` para evitar pérdida accidental de datos o resultados.
    """
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        resolved_root,
        resolved_root / "data",
        resolved_root / "models",
        resolved_root / "results",
    }
    if not _is_relative_to(resolved_output, resolved_root):
        raise ValueError("--overwrite solo admite carpetas de salida dentro del repositorio.")
    if resolved_output in dangerous_paths:
        raise ValueError(f"Se rechazó una ruta de salida peligrosa: {resolved_output}")

    resolved_output.mkdir(parents=True, exist_ok=True)
    for child in resolved_output.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def estimate_thumbnail_tissue_mask(rgb_image: Image.Image | np.ndarray) -> np.ndarray:
    """
    ***
    * rgb_image: Imagen RGB o arreglo con tres canales.
    ***
    Estima tejido con la regla original `mean < 235` y `std > 8`.
    Retorna una máscara booleana sin modificar la imagen.
    """
    array = np.asarray(rgb_image.convert("RGB") if isinstance(rgb_image, Image.Image) else rgb_image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("rgb_image debe tener forma (alto, ancho, 3).")
    rgb = array[..., :3].astype(np.float32)
    return (np.mean(rgb, axis=-1) < 235) & (np.std(rgb, axis=-1) > 8)


def compute_simple_tissue_ratio(rgb_image: Image.Image | np.ndarray) -> float:
    """Calcula la fracción aproximada de tejido usando la máscara RGB original."""
    tissue_mask = estimate_thumbnail_tissue_mask(rgb_image)
    return float(np.mean(tissue_mask)) if tissue_mask.size else 0.0


def _slide_property(slide: object, property_name: str) -> str | None:
    """Lee una propiedad OpenSlide opcional como texto."""
    value = getattr(slide, "properties", {}).get(property_name)
    return None if value is None else str(value)
