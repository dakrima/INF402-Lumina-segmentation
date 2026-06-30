"""Generación de previews para los patches seleccionados."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.visualization.patch_preview import PatchBox, save_patch_selection_preview


def _row_to_patch_box(
    row: dict[str, object],
    slide_dimensions: tuple[int, int],
    thumbnail_dimensions: tuple[int, int],
) -> PatchBox:
    slide_width, slide_height = slide_dimensions
    thumb_width, thumb_height = thumbnail_dimensions
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height

    x_level0 = int(row["x_level0"])
    y_level0 = int(row["y_level0"])
    patch_size = int(row["patch_size"])

    return PatchBox(
        x=int(round(x_level0 * scale_x)),
        y=int(round(y_level0 * scale_y)),
        width=max(1, int(round(patch_size * scale_x))),
        height=max(1, int(round(patch_size * scale_y))),
        selected=bool(row["selected"]),
    )


def save_wsi_patch_selection_preview(
    thumbnail: Image.Image,
    candidate_rows: list[dict[str, object]],
    slide_dimensions: tuple[int, int],
    output_path: Path,
) -> Path:
    """
    ***
    * thumbnail: Vista reducida RGB de la WSI.
    * candidate_rows: Candidatos evaluados con coordenadas y estado de selección.
    * slide_dimensions: Dimensiones originales de la WSI.
    * output_path: Ruta de la preview.
    ***
    Escala las coordenadas al thumbnail, dibuja los rectángulos y guarda la imagen.

    Retorna la ruta de la preview generada.
    """
    boxes = [
        _row_to_patch_box(
            row=row,
            slide_dimensions=slide_dimensions,
            thumbnail_dimensions=thumbnail.size,
        )
        for row in candidate_rows
    ]
    return save_patch_selection_preview(
        rgb_image=thumbnail,
        patches=boxes,
        output_path=output_path,
    )
