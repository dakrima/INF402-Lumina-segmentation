"""Proxies clásicos y deterministas de calidad y utilidad para patches H&E."""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np
from PIL import Image

from src.preprocessing.wsi_patch_extraction import estimate_thumbnail_tissue_mask


MEDICAL_STAIN_FEATURE_FIELDS = [
    "stain_contrast_score",
    "stain_balance_score",
    "hematoxylin_dynamic_range",
    "eosin_dynamic_range",
    "stain_saturation_penalty",
    "low_contrast_penalty",
]

MEDICAL_TISSUE_FEATURE_FIELDS = [
    "clean_tissue_ratio",
    "foreground_component_score",
    "border_background_ratio",
    "border_tissue_penalty",
    "fat_background_penalty",
]

MEDICAL_TEXTURE_FEATURE_FIELDS = [
    "texture_entropy_score",
    "local_entropy_score",
    "gradient_energy_score",
    "laplacian_texture_score",
    "multi_scale_entropy_score",
    "glcm_contrast_proxy",
    "glcm_homogeneity_proxy",
]

MEDICAL_SHARPNESS_FEATURE_FIELDS = [
    "tenengrad_sharpness_score",
    "laplacian_sharpness_score",
    "sharpness_quality_score",
    "fold_or_edge_artifact_penalty",
    "extreme_intensity_penalty",
]

MEDICAL_PSEUDO_CELLULARITY_FEATURE_FIELDS = [
    "pseudo_nuclear_density_score",
    "pseudo_nuclear_component_count",
    "pseudo_nuclear_component_density",
    "pseudo_nuclear_size_variability",
    "pseudo_nuclear_clustering_score",
]

MEDICAL_SCORE_FIELDS = [
    "medical_stain_quality_score",
    "medical_tissue_quality_score",
    "medical_texture_score",
    "medical_sharpness_score",
    "medical_pseudo_cellularity_score",
    "medical_artifact_penalty",
    "medical_image_quality_score",
    "medical_image_utility_score",
]

MEDICAL_IMAGE_FEATURE_FIELDS = [
    *MEDICAL_STAIN_FEATURE_FIELDS,
    *MEDICAL_TISSUE_FEATURE_FIELDS,
    *MEDICAL_TEXTURE_FEATURE_FIELDS,
    *MEDICAL_SHARPNESS_FEATURE_FIELDS,
    *MEDICAL_PSEUDO_CELLULARITY_FEATURE_FIELDS,
    *MEDICAL_SCORE_FIELDS,
]

MEDICAL_IMAGE_FEATURE_METHODS = {
    "stain": "fixed_he_optical_density_dynamic_range_v1",
    "tissue": "thumbnail_tissue_mask_border_component_proxy_v1",
    "texture": "entropy_gradient_laplacian_glcm_proxy_v1",
    "sharpness": "tenengrad_laplacian_extreme_intensity_proxy_v1",
    "pseudo_cellularity": "hematoxylin_threshold_component_proxy_v1",
    "scores": "weighted_classical_medical_image_proxy_v1",
}


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _resize_for_features(rgb_image: Image.Image, feature_size: int) -> Image.Image:
    if feature_size <= 0:
        raise ValueError("feature_size debe ser mayor que cero.")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return rgb_image.convert("RGB").resize((feature_size, feature_size), resampling)


def _rgb_array_01(rgb_image: Image.Image) -> np.ndarray:
    return np.asarray(rgb_image.convert("RGB"), dtype=np.float32) / 255.0


def _gray(rgb_array: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb_array[..., 0]
        + 0.587 * rgb_array[..., 1]
        + 0.114 * rgb_array[..., 2]
    ).astype(np.float32, copy=False)


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return values.reshape(-1)
    return values[mask]


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(np.sum(hist))
    if total <= 0:
        return 0.0
    probabilities = hist[hist > 0].astype(np.float64) / total
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    max_entropy = float(np.log2(bins))
    return _clip01(entropy / max_entropy if max_entropy > 0 else 0.0)


def _he_channels(rgb_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-6
    optical_density = -np.log(np.clip(rgb_array, eps, 1.0))
    stain_matrix = np.array(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ],
        dtype=np.float32,
    )
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    concentrations = optical_density.reshape(-1, 3) @ np.linalg.pinv(stain_matrix)
    hematoxylin = np.maximum(0.0, concentrations[:, 0]).reshape(rgb_array.shape[:2])
    eosin = np.maximum(0.0, concentrations[:, 1]).reshape(rgb_array.shape[:2])
    return hematoxylin.astype(np.float32), eosin.astype(np.float32)


def _dynamic_range(values: np.ndarray, scale: float = 1.2) -> float:
    if values.size == 0:
        return 0.0
    p95 = float(np.percentile(values, 95))
    p05 = float(np.percentile(values, 5))
    return _clip01((p95 - p05) / scale)


def _small_mask(mask: np.ndarray, max_size: int = 128) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(bool)
    height, width = mask.shape
    scale = max(height, width) / max_size
    if scale <= 1:
        return mask.astype(bool)
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = pil.resize(
        (max(1, int(round(width / scale))), max(1, int(round(height / scale)))),
        getattr(Image, "Resampling", Image).NEAREST,
    )
    return np.asarray(resized) > 0


def _component_sizes(mask: np.ndarray, max_size: int = 128) -> list[int]:
    """Retorna tamaños de componentes conectados en una máscara reducida."""
    small = _small_mask(mask, max_size=max_size)
    height, width = small.shape
    visited = np.zeros_like(small, dtype=bool)
    sizes: list[int] = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not small[row, col]:
                continue
            visited[row, col] = True
            queue: deque[tuple[int, int]] = deque([(row, col)])
            size = 0
            while queue:
                current_row, current_col = queue.popleft()
                size += 1
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and small[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            sizes.append(size)
    return sizes


def _foreground_component_score(mask: np.ndarray) -> float:
    sizes = _component_sizes(mask, max_size=96)
    if not sizes:
        return 0.0
    total = float(sum(sizes))
    largest_fraction = max(sizes) / total if total > 0 else 0.0
    component_penalty = min(1.0, max(0, len(sizes) - 1) / 12.0)
    return _clip01(0.75 * largest_fraction + 0.25 * (1.0 - component_penalty))


def _border_values(mask: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
    height, width = mask.shape
    border = max(1, int(round(min(height, width) * border_fraction)))
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    return mask[border_mask]


def _tile_values(values: np.ndarray, tile_count: int = 8) -> Iterable[np.ndarray]:
    height, width = values.shape
    tile_height = max(1, height // tile_count)
    tile_width = max(1, width // tile_count)
    for row in range(0, height, tile_height):
        for col in range(0, width, tile_width):
            yield values[row:row + tile_height, col:col + tile_width]


def _local_entropy(gray: np.ndarray) -> float:
    entropies = [_entropy(tile.reshape(-1), bins=16) for tile in _tile_values(gray, 8)]
    return _clip01(float(np.mean(entropies)) if entropies else 0.0)


def _multi_scale_entropy(gray: np.ndarray) -> float:
    entropies = [_entropy(gray.reshape(-1), bins=32)]
    for scale in (0.5, 0.25):
        height = max(2, int(round(gray.shape[0] * scale)))
        width = max(2, int(round(gray.shape[1] * scale)))
        image = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
        resized = image.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)
        entropies.append(_entropy(np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0, bins=32))
    return _clip01(float(np.mean(entropies)))


def _gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dy, dx = np.gradient(gray.astype(np.float32))
    return dx.astype(np.float32), dy.astype(np.float32)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )


def _glcm_proxy(gray: np.ndarray, mask: np.ndarray, bins: int = 16) -> tuple[float, float]:
    quantized = np.clip((gray * bins).astype(np.int16), 0, bins - 1)
    pairs: list[tuple[np.ndarray, np.ndarray]] = [
        (quantized[:, :-1], quantized[:, 1:]),
        (quantized[:-1, :], quantized[1:, :]),
    ]
    if mask.shape == gray.shape and np.any(mask):
        valid_masks = [
            mask[:, :-1] & mask[:, 1:],
            mask[:-1, :] & mask[1:, :],
        ]
    else:
        valid_masks = [np.ones_like(pairs[0][0], dtype=bool), np.ones_like(pairs[1][0], dtype=bool)]

    contrast_values: list[float] = []
    homogeneity_values: list[float] = []
    for (left, right), valid in zip(pairs, valid_masks):
        if not np.any(valid):
            continue
        diff = np.abs(left[valid] - right[valid]).astype(np.float32)
        contrast_values.append(float(np.mean((diff / (bins - 1)) ** 2)))
        homogeneity_values.append(float(np.mean(1.0 / (1.0 + diff))))
    contrast = float(np.mean(contrast_values)) if contrast_values else 0.0
    homogeneity = float(np.mean(homogeneity_values)) if homogeneity_values else 0.0
    return _clip01(contrast * 4.0), _clip01(homogeneity)


def _pseudo_nuclear_features(hematoxylin: np.ndarray, tissue_mask: np.ndarray) -> dict[str, float]:
    values = _masked_values(hematoxylin, tissue_mask)
    if values.size == 0:
        return {
            "pseudo_nuclear_density_score": 0.0,
            "pseudo_nuclear_component_count": 0.0,
            "pseudo_nuclear_component_density": 0.0,
            "pseudo_nuclear_size_variability": 0.0,
            "pseudo_nuclear_clustering_score": 0.0,
        }
    threshold = max(float(np.percentile(values, 70)), float(np.mean(values) + 0.25 * np.std(values)))
    pseudo_mask = (hematoxylin >= threshold) & tissue_mask
    density = float(np.mean(pseudo_mask[tissue_mask])) if np.any(tissue_mask) else float(np.mean(pseudo_mask))
    sizes = _component_sizes(pseudo_mask, max_size=96)
    component_count = float(len(sizes))
    tissue_pixels_small = max(1, int(np.sum(_small_mask(tissue_mask, max_size=96))))
    component_density = _clip01(component_count / tissue_pixels_small * 160.0)
    if len(sizes) > 1:
        size_mean = float(np.mean(sizes))
        size_variability = _clip01(float(np.std(sizes)) / (size_mean + 1e-6))
    else:
        size_variability = 0.0
    tile_densities = [
        float(np.mean(tile))
        for tile in _tile_values(pseudo_mask.astype(np.float32), 8)
        if tile.size
    ]
    clustering = _clip01(float(np.std(tile_densities)) * 4.0) if tile_densities else 0.0
    return {
        "pseudo_nuclear_density_score": _clip01(density * 4.0),
        "pseudo_nuclear_component_count": component_count,
        "pseudo_nuclear_component_density": component_density,
        "pseudo_nuclear_size_variability": size_variability,
        "pseudo_nuclear_clustering_score": clustering,
    }


def _score_medical_features(features: dict[str, float], tumor_bed_relevance_proxy: float) -> dict[str, float]:
    stain_quality = _clip01(
        0.35 * features["stain_contrast_score"]
        + 0.25 * features["stain_balance_score"]
        + 0.20 * features["hematoxylin_dynamic_range"]
        + 0.20 * features["eosin_dynamic_range"]
        - 0.20 * features["stain_saturation_penalty"]
        - 0.15 * features["low_contrast_penalty"]
    )
    tissue_quality = _clip01(
        0.50 * features["clean_tissue_ratio"]
        + 0.30 * features["foreground_component_score"]
        + 0.20 * (1.0 - features["border_tissue_penalty"])
        - 0.15 * features["fat_background_penalty"]
    )
    texture_score = _clip01(
        0.25 * features["texture_entropy_score"]
        + 0.20 * features["local_entropy_score"]
        + 0.20 * features["gradient_energy_score"]
        + 0.15 * features["laplacian_texture_score"]
        + 0.10 * features["multi_scale_entropy_score"]
        + 0.10 * features["glcm_contrast_proxy"]
    )
    sharpness_score = _clip01(
        0.45 * features["tenengrad_sharpness_score"]
        + 0.35 * features["laplacian_sharpness_score"]
        + 0.20 * (1.0 - features["fold_or_edge_artifact_penalty"])
    )
    pseudo_cellularity = _clip01(
        0.45 * features["pseudo_nuclear_density_score"]
        + 0.30 * features["pseudo_nuclear_component_density"]
        + 0.15 * (1.0 - features["pseudo_nuclear_size_variability"])
        + 0.10 * features["pseudo_nuclear_clustering_score"]
    )
    artifact_penalty = _clip01(
        0.30 * features["stain_saturation_penalty"]
        + 0.25 * features["low_contrast_penalty"]
        + 0.20 * features["border_tissue_penalty"]
        + 0.15 * features["fold_or_edge_artifact_penalty"]
        + 0.10 * features["extreme_intensity_penalty"]
    )
    image_quality = _clip01(
        0.25 * stain_quality
        + 0.25 * tissue_quality
        + 0.20 * sharpness_score
        + 0.20 * texture_score
        - 0.10 * artifact_penalty
    )
    image_utility = _clip01(
        0.35 * image_quality
        + 0.25 * texture_score
        + 0.20 * pseudo_cellularity
        + 0.20 * _clip01(tumor_bed_relevance_proxy)
    )
    return {
        "medical_stain_quality_score": stain_quality,
        "medical_tissue_quality_score": tissue_quality,
        "medical_texture_score": texture_score,
        "medical_sharpness_score": sharpness_score,
        "medical_pseudo_cellularity_score": pseudo_cellularity,
        "medical_artifact_penalty": artifact_penalty,
        "medical_image_quality_score": image_quality,
        "medical_image_utility_score": image_utility,
    }


def compute_medical_image_features(
    rgb_patch: Image.Image,
    *,
    feature_size: int = 512,
    tumor_bed_relevance_proxy: float = 0.0,
) -> dict[str, float]:
    """
    ***
    * rgb_patch: Imagen RGB del candidato.
    * feature_size: Tamaño de trabajo para las features clásicas.
    * tumor_bed_relevance_proxy: Score técnico heredado que participa en utilidad.
    ***
    Calcula proxies de tinción, tejido, textura, nitidez, pseudo-celularidad y
    artefactos. Retorna features y scores médicos técnicos normalizados.
    """
    feature_patch = _resize_for_features(rgb_patch, feature_size=feature_size)
    rgb_array = _rgb_array_01(feature_patch)
    gray = _gray(rgb_array)
    tissue_mask = estimate_thumbnail_tissue_mask(feature_patch)
    clean_tissue_ratio = float(np.mean(tissue_mask)) if tissue_mask.size else 0.0
    tissue_gray = _masked_values(gray, tissue_mask)
    hematoxylin, eosin = _he_channels(rgb_array)
    h_values = _masked_values(hematoxylin, tissue_mask)
    e_values = _masked_values(eosin, tissue_mask)

    brightness = np.mean(rgb_array, axis=-1)
    saturation = np.max(rgb_array, axis=-1) - np.min(rgb_array, axis=-1)
    stain_contrast_score = _dynamic_range(tissue_gray, scale=0.55)
    hematoxylin_dynamic_range = _dynamic_range(h_values, scale=1.2)
    eosin_dynamic_range = _dynamic_range(e_values, scale=1.2)
    stain_balance_score = _clip01(
        1.0
        - abs(hematoxylin_dynamic_range - eosin_dynamic_range)
        / (hematoxylin_dynamic_range + eosin_dynamic_range + 1e-6)
    )
    stain_saturation_penalty = _clip01(
        float(np.mean((saturation > 0.82) | ((saturation < 0.03) & (brightness > 0.85)))) * 3.0
    )
    low_contrast_penalty = _clip01(1.0 - stain_contrast_score)

    border_background_ratio = float(np.mean(~_border_values(tissue_mask))) if tissue_mask.size else 1.0
    border_tissue_penalty = _clip01(max(0.0, border_background_ratio - 0.30) / 0.70)
    fat_background_penalty = _clip01(
        float(np.mean((brightness > 0.78) & (saturation < 0.12) & (~tissue_mask))) * 3.0
    )
    foreground_component_score = _foreground_component_score(tissue_mask)

    texture_entropy_score = _entropy(tissue_gray, bins=32)
    local_entropy_score = _local_entropy(gray)
    multi_scale_entropy_score = _multi_scale_entropy(gray)
    dx, dy = _gradient(gray)
    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
    gradient_energy_score = _clip01(float(np.mean(gradient_magnitude)) * 8.0)
    laplacian = _laplacian(gray)
    laplacian_texture_score = _clip01(float(np.mean(np.abs(laplacian))) * 8.0)
    glcm_contrast_proxy, glcm_homogeneity_proxy = _glcm_proxy(gray, tissue_mask)

    tenengrad_sharpness_score = _clip01(float(np.mean(gradient_magnitude ** 2)) * 80.0)
    laplacian_sharpness_score = _clip01(float(np.var(laplacian)) * 80.0)
    fold_or_edge_artifact_penalty = _clip01(float(np.mean(gradient_magnitude > 0.35)) * 4.0)
    extreme_intensity_penalty = _clip01(
        float(np.mean((brightness < 0.03) | (brightness > 0.97))) * 4.0
    )

    features = {
        "stain_contrast_score": stain_contrast_score,
        "stain_balance_score": stain_balance_score,
        "hematoxylin_dynamic_range": hematoxylin_dynamic_range,
        "eosin_dynamic_range": eosin_dynamic_range,
        "stain_saturation_penalty": stain_saturation_penalty,
        "low_contrast_penalty": low_contrast_penalty,
        "clean_tissue_ratio": _clip01(clean_tissue_ratio),
        "foreground_component_score": foreground_component_score,
        "border_background_ratio": _clip01(border_background_ratio),
        "border_tissue_penalty": border_tissue_penalty,
        "fat_background_penalty": fat_background_penalty,
        "texture_entropy_score": texture_entropy_score,
        "local_entropy_score": local_entropy_score,
        "gradient_energy_score": gradient_energy_score,
        "laplacian_texture_score": laplacian_texture_score,
        "multi_scale_entropy_score": multi_scale_entropy_score,
        "glcm_contrast_proxy": glcm_contrast_proxy,
        "glcm_homogeneity_proxy": glcm_homogeneity_proxy,
        "tenengrad_sharpness_score": tenengrad_sharpness_score,
        "laplacian_sharpness_score": laplacian_sharpness_score,
        "sharpness_quality_score": _clip01(
            0.50 * tenengrad_sharpness_score + 0.50 * laplacian_sharpness_score
        ),
        "fold_or_edge_artifact_penalty": fold_or_edge_artifact_penalty,
        "extreme_intensity_penalty": extreme_intensity_penalty,
        **_pseudo_nuclear_features(hematoxylin, tissue_mask),
    }
    features.update(
        _score_medical_features(
            features,
            tumor_bed_relevance_proxy=tumor_bed_relevance_proxy,
        )
    )
    return {
        field_name: _clip01(value) if field_name != "pseudo_nuclear_component_count" else float(value)
        for field_name, value in features.items()
    }
