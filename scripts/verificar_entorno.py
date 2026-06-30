#!/usr/bin/env python
"""Verifica las dependencias y carpetas mínimas del experimento final."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("NumPy", "numpy"),
    ("Pillow", "PIL"),
    ("OpenSlide", "openslide"),
    ("TIAToolbox", "tiatoolbox"),
    ("PyTorch", "torch"),
    ("timm", "timm"),
    ("scikit-learn", "sklearn"),
    ("psutil", "psutil"),
]
REQUIRED_DIRECTORIES = ["data", "results", "src", "scripts"]


def _module_version(module: object) -> str:
    return str(getattr(module, "__version__", "versión no informada"))


def check_imports() -> tuple[list[str], dict[str, object]]:
    """Importa cada dependencia y retorna los nombres que fallaron."""
    failed: list[str] = []
    imported: dict[str, object] = {}
    print("\nDependencias")
    print("------------")
    for display_name, import_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001 - diagnóstico de dependencias
            failed.append(display_name)
            print(f"[ERROR] {display_name}: {exc}")
            continue
        imported[display_name] = module
        print(f"[ OK ] {display_name}: {_module_version(module)}")
    return failed, imported


def check_device(imported: dict[str, object]) -> None:
    """Informa los dispositivos disponibles; el experimento original usa CPU."""
    torch_module = imported.get("PyTorch")
    if torch_module is None:
        return
    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = bool(
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    )
    print(f"\nDispositivos: CPU disponible, CUDA={cuda_available}, MPS={mps_available}")


def check_directories(root_dir: Path) -> list[str]:
    """Retorna las carpetas mínimas ausentes en la raíz del proyecto."""
    missing = [path for path in REQUIRED_DIRECTORIES if not (root_dir / path).is_dir()]
    print("\nCarpetas")
    print("--------")
    for path in REQUIRED_DIRECTORIES:
        print(f"[{'ERROR' if path in missing else ' OK '}] {path}/")
    return missing


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    print("Verificación del entorno INF402 Lumina")
    print("======================================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Ejecutable: {sys.executable}")
    print(f"Plataforma: {platform.platform()}")
    failed_imports, imported = check_imports()
    check_device(imported)
    missing_directories = check_directories(root_dir)
    if missing_directories or failed_imports:
        print("\nEl entorno está incompleto.")
        return 1
    print("\nEl entorno contiene todas las dependencias mínimas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
