"""Extracción, cache, clustering y distancias de embeddings UNI."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.selection.manifests import utc_now_iso


UNI_BACKEND_MISSING_MESSAGE = (
    "UNI embedding backend requested but model weights/path are not available. "
    "Provide --embedding-model-path or configure access to the UNI model."
)


@dataclass(frozen=True)
class EmbeddingExtractorConfig:
    """Configuración del extractor de embeddings."""

    embedding_backend: str = "uni"
    embedding_model_name: str = "UNI"
    embedding_model_path: Path | None = None
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_num_workers: int = 2
    embedding_dim: int | None = None
    embedding_distance_metric: str = "cosine"


class PatchEmbeddingExtractor:
    """Wrapper de PyTorch que retorna un embedding por patch."""

    def __init__(
        self,
        *,
        model: object,
        torch_module: object,
        device: object,
        model_name: str,
        backend: str,
        model_path: Path,
        distance_metric: str,
    ) -> None:
        self.model = model
        self.torch = torch_module
        self.device = device
        self.model_name = model_name
        self.backend = backend
        self.model_path = model_path
        self.distance_metric = distance_metric
        self._inference_lock = threading.Lock()
        self._wait_stats_lock = threading.Lock()
        self._wait_seconds_by_thread: dict[int, float] = {}

    def _preprocess_batch(self, patches: Sequence[Image.Image]) -> object:
        tensors = [
            _pil_to_imagenet_tensor(patch, torch_module=self.torch)
            for patch in patches
        ]
        return self.torch.stack(tensors, dim=0).to(self.device)

    def embed_batch(self, patches: Sequence[Image.Image]) -> np.ndarray:
        """Retorna los embeddings de un lote de patches RGB."""
        if not patches:
            return np.zeros((0, 0), dtype=np.float32)
        wait_started = time.perf_counter()
        with self._inference_lock:
            wait_seconds = time.perf_counter() - wait_started
            thread_id = threading.get_ident()
            with self._wait_stats_lock:
                self._wait_seconds_by_thread[thread_id] = (
                    self._wait_seconds_by_thread.get(thread_id, 0.0) + wait_seconds
                )
            batch = self._preprocess_batch(patches)
            with self.torch.no_grad():
                output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            for key in ("embedding", "embeddings", "features", "x"):
                if key in output:
                    output = output[key]
                    break
        if hasattr(output, "detach"):
            output = output.detach().cpu().float().numpy()
        embeddings = np.asarray(output, dtype=np.float32)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Embedding model returned shape {embeddings.shape}; expected 2D array."
            )
        if self.distance_metric == "cosine":
            embeddings = normalize_embeddings(embeddings)
        return embeddings.astype(np.float32, copy=False)

    def current_thread_wait_seconds(self) -> float:
        """Retorna el tiempo acumulado esperando el modelo UNI compartido."""
        with self._wait_stats_lock:
            return float(self._wait_seconds_by_thread.get(threading.get_ident(), 0.0))


def _import_torch() -> object:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic
        raise RuntimeError(
            "UNI embedding backend requires PyTorch. Activate the project environment "
            "and provide --embedding-model-path."
        ) from exc
    return torch


def _resolve_device(torch_module: object, requested_device: str) -> object:
    if requested_device == "auto":
        if torch_module.cuda.is_available():
            requested_device = "cuda"
        elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Embedding device cuda was requested but CUDA is not available.")
    if requested_device == "mps" and not (
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    ):
        raise RuntimeError("Embedding device mps was requested but MPS is not available.")
    return torch_module.device(requested_device)


def _pil_to_imagenet_tensor(patch: Image.Image, *, torch_module: object) -> object:
    """Convierte un patch a tensor ImageNet normalizado de 224x224."""
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = patch.convert("RGB").resize((224, 224), resampling)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    return torch_module.from_numpy(array.transpose(2, 0, 1)).float()


def _checkpoint_state_dict(checkpoint: object) -> dict[str, object]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "teacher", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("UNI checkpoint did not contain a loadable state_dict.")
    cleaned: dict[str, object] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue
        normalized_key = key
        for prefix in ("module.", "model.", "backbone."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        cleaned[normalized_key] = value
    return cleaned


def _load_uni_model_from_path(
    model_path: Path,
    *,
    torch_module: object,
    device: object,
) -> object:
    if not model_path.exists():
        raise RuntimeError(f"{UNI_BACKEND_MISSING_MESSAGE} Path does not exist: {model_path}")

    if model_path.is_dir():
        preferred_names = [
            "pytorch_model.bin",
            "model.pth",
            "model.pt",
            "checkpoint.pth",
            "checkpoint.pt",
        ]
        for name in preferred_names:
            candidate = model_path / name
            if candidate.exists():
                model_path = candidate
                break
        else:
            raise RuntimeError(
                "UNI model path is a directory, but no supported checkpoint file was found. "
                "Provide --embedding-model-path pointing to a local .pt/.pth/.bin checkpoint "
                "or TorchScript file."
            )

    suffix = model_path.suffix.lower()
    if suffix in {".jit", ".ts", ".torchscript"}:
        model = torch_module.jit.load(str(model_path), map_location=device)
        model.to(device)
        model.eval()
        return model

    try:
        import timm  # type: ignore
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic
        raise RuntimeError(
            "Loading a UNI checkpoint requires timm unless the file is TorchScript. "
            "Install/activate timm in the project environment or provide a TorchScript model."
        ) from exc

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    checkpoint = torch_module.load(str(model_path), map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 80:
        raise RuntimeError(
            "UNI checkpoint could not be loaded into the expected ViT-L/16 backbone. "
            "Check --embedding-model-path and model format."
        )
    if unexpected and len(unexpected) > 80:
        raise RuntimeError(
            "UNI checkpoint has too many unexpected keys for the expected ViT-L/16 backbone."
        )
    model.to(device)
    model.eval()
    return model


def _build_embedding_extractor(config: EmbeddingExtractorConfig) -> PatchEmbeddingExtractor:
    if config.embedding_backend != "uni":
        raise RuntimeError(
            f"Unsupported embedding backend '{config.embedding_backend}'. Only 'uni' is implemented."
        )
    if config.embedding_model_path is None:
        raise RuntimeError(UNI_BACKEND_MISSING_MESSAGE)
    if config.embedding_distance_metric != "cosine":
        raise RuntimeError("Only cosine embedding distance is currently implemented.")

    torch_module = _import_torch()
    device = _resolve_device(torch_module, config.embedding_device)
    model = _load_uni_model_from_path(
        config.embedding_model_path.expanduser().resolve(),
        torch_module=torch_module,
        device=device,
    )
    return PatchEmbeddingExtractor(
        model=model,
        torch_module=torch_module,
        device=device,
        model_name=config.embedding_model_name,
        backend=config.embedding_backend,
        model_path=config.embedding_model_path.expanduser().resolve(),
        distance_metric=config.embedding_distance_metric,
    )


_EMBEDDING_EXTRACTOR_CACHE_LOCK = threading.Lock()
_EMBEDDING_EXTRACTOR_CACHE: tuple[tuple[str, ...], PatchEmbeddingExtractor] | None = None
_EMBEDDING_EXTRACTOR_LOAD_COUNT = 0


def _embedding_extractor_cache_key(config: EmbeddingExtractorConfig) -> tuple[str, ...]:
    model_path = (
        str(config.embedding_model_path.expanduser().resolve())
        if config.embedding_model_path is not None
        else ""
    )
    return (
        config.embedding_backend,
        config.embedding_model_name,
        model_path,
        config.embedding_device,
        config.embedding_distance_metric,
    )


def build_embedding_extractor(config: EmbeddingExtractorConfig) -> PatchEmbeddingExtractor:
    """Retorna una única instancia compartida del modelo UNI configurado."""
    global _EMBEDDING_EXTRACTOR_CACHE, _EMBEDDING_EXTRACTOR_LOAD_COUNT
    key = _embedding_extractor_cache_key(config)
    with _EMBEDDING_EXTRACTOR_CACHE_LOCK:
        if _EMBEDDING_EXTRACTOR_CACHE is not None and _EMBEDDING_EXTRACTOR_CACHE[0] == key:
            return _EMBEDDING_EXTRACTOR_CACHE[1]
        extractor = _build_embedding_extractor(config)
        _EMBEDDING_EXTRACTOR_CACHE = (key, extractor)
        _EMBEDDING_EXTRACTOR_LOAD_COUNT += 1
        return extractor


def embedding_extractor_load_count() -> int:
    """Retorna cuántas veces se cargó UNI en el proceso."""
    with _EMBEDDING_EXTRACTOR_CACHE_LOCK:
        return _EMBEDDING_EXTRACTOR_LOAD_COUNT


def compute_patch_embeddings(
    extractor: PatchEmbeddingExtractor,
    patches: Sequence[Image.Image],
) -> np.ndarray:
    """Calcula embeddings para una secuencia de patches."""
    return extractor.embed_batch(patches)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normaliza cada embedding por L2 para calcular distancia coseno."""
    if embeddings.size == 0:
        return embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return (embeddings / norms).astype(np.float32, copy=False)


def cosine_distance_to_selected(
    candidate_embedding: np.ndarray,
    selected_embeddings: np.ndarray,
) -> tuple[float, float]:
    """Retorna distancia coseno mínima y similitud máxima a los seleccionados."""
    if selected_embeddings.size == 0:
        return 1.0, 0.0
    similarities = selected_embeddings @ candidate_embedding
    max_similarity = float(np.max(similarities))
    min_distance = float(1.0 - max_similarity)
    return max(0.0, min(2.0, min_distance)), max(-1.0, min(1.0, max_similarity))


def write_embedding_cache(
    *,
    embeddings: np.ndarray,
    candidate_ids: list[str],
    cache_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Guarda embeddings y metadata en archivos de cache de la corrida."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32, copy=False),
        candidate_ids=np.array(candidate_ids, dtype=object),
    )
    payload = {
        **metadata,
        "candidate_ids": candidate_ids,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "num_embeddings": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        "created_at": metadata.get("created_at") or utc_now_iso(),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache_path, metadata_path


def load_embedding_cache(
    *,
    cache_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Carga un cache generado por `write_embedding_cache`."""
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path} / {metadata_path}"
        )
    with np.load(cache_path, allow_pickle=True) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        candidate_ids = [str(value) for value in data["candidate_ids"].tolist()]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return embeddings, candidate_ids, metadata


def validate_embedding_cache(
    *,
    candidate_ids: list[str],
    embeddings: np.ndarray,
    cached_candidate_ids: list[str],
    metadata: dict[str, Any],
    embedding_backend: str,
    embedding_model_name: str,
    embedding_distance_metric: str,
    expected_dim: int | None,
) -> str | None:
    """Retorna `None` si el cache coincide; de lo contrario explica la diferencia."""
    if candidate_ids != cached_candidate_ids:
        return "Embedding cache candidate_ids do not match the current candidate subset."
    if metadata.get("embedding_backend") != embedding_backend:
        return "Embedding cache was created with a different embedding_backend."
    if metadata.get("embedding_model_name") != embedding_model_name:
        return "Embedding cache was created with a different embedding_model_name."
    if metadata.get("embedding_distance_metric") != embedding_distance_metric:
        return "Embedding cache was created with a different distance metric."
    if embeddings.ndim != 2 or embeddings.shape[0] != len(candidate_ids):
        return "Embedding cache has invalid shape."
    if expected_dim is not None and int(embeddings.shape[1]) != expected_dim:
        return "Embedding cache dimension does not match --embedding-dim."
    return None


def _embedding_distance(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return 1.0 - (a @ b.T)
    if metric == "euclidean":
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.sqrt(np.maximum(squared, 0.0))
    raise ValueError(f"Unsupported embedding distance metric: {metric}")


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    cluster_count: int,
    seed: int,
    distance_metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Agrupa embeddings y retorna etiquetas, centroides, método y advertencias."""
    warnings: list[str] = []
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    n_rows = embeddings.shape[0]
    if n_rows == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 0), dtype=np.float32), "none", warnings
    k = max(1, min(int(cluster_count), n_rows))
    if k < cluster_count:
        warnings.append(
            f"embedding_cluster_count reduced from {cluster_count} to {k} because fewer embeddings are available."
        )

    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency fallback
        labels, centroids = _fallback_kmeans(
            embeddings,
            cluster_count=k,
            seed=seed,
            distance_metric=distance_metric,
        )
        return labels, centroids, "fallback_kmeans", warnings

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings).astype(np.int64)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    if distance_metric == "cosine":
        centroids = normalize_embeddings(centroids)
    return labels, centroids, "sklearn_kmeans", warnings


def _fallback_kmeans(
    embeddings: np.ndarray,
    *,
    cluster_count: int,
    seed: int,
    distance_metric: str,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_rows = embeddings.shape[0]
    first_index = int(rng.integers(0, n_rows))
    centroid_indices = [first_index]
    while len(centroid_indices) < cluster_count:
        centroids = embeddings[centroid_indices]
        distances = _embedding_distance(embeddings, centroids, distance_metric)
        min_distances = np.min(distances, axis=1)
        next_index = int(np.argmax(min_distances))
        if next_index in centroid_indices:
            break
        centroid_indices.append(next_index)
    while len(centroid_indices) < cluster_count:
        for index in range(n_rows):
            if index not in centroid_indices:
                centroid_indices.append(index)
                break

    centroids = embeddings[centroid_indices].astype(np.float32, copy=True)
    labels = np.zeros((n_rows,), dtype=np.int64)
    for _ in range(iterations):
        distances = _embedding_distance(embeddings, centroids, distance_metric)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        new_centroids = []
        for cluster_id in range(cluster_count):
            members = embeddings[labels == cluster_id]
            if members.size == 0:
                new_centroids.append(centroids[cluster_id])
            else:
                new_centroids.append(np.mean(members, axis=0))
        centroids = np.asarray(new_centroids, dtype=np.float32)
        if distance_metric == "cosine":
            centroids = normalize_embeddings(centroids)
    return labels, centroids


def embedding_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    distance_metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna distancia al centroide y score de representatividad."""
    if embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    distances = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for index, label in enumerate(labels):
        centroid = centroids[int(label)][None, :]
        distances[index] = float(_embedding_distance(embeddings[index:index + 1], centroid, distance_metric)[0, 0])
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return distances, np.zeros_like(distances)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        representativeness = np.ones_like(distances, dtype=np.float32)
    else:
        representativeness = 1.0 - ((distances - minimum) / (maximum - minimum))
        representativeness = np.clip(representativeness, 0.0, 1.0).astype(np.float32)
    return distances.astype(np.float32), representativeness
