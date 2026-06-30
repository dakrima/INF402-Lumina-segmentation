"""Estructura mínima de un candidato de patch."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchCandidate:
    """Coordenada de nivel 0 con proporción de tejido estimada en thumbnail."""

    candidate_id: str
    grid_index: int
    x_level0: int
    y_level0: int
    patch_size: int
    thumbnail_tissue_ratio: float
