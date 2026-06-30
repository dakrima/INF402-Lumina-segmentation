#!/usr/bin/env python
"""Verifica las dependencias y carpetas mínimas del experimento final."""

from __future__ import annotations

import argparse
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
    """Retorna la versión informada por un módulo o un texto de respaldo."""
    return str(getattr(module, "__version__", "versión no informada"))


def check_imports() -> tuple[list[str], dict[str, object]]:
    """
    Importa cada dependencia requerida por el pipeline.

    Retorna los nombres que fallaron y los módulos importados correctamente.
    """
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
    """
    ***
    * imported: Módulos cargados por `check_imports`.
    ***
    Informa la disponibilidad de CPU, CUDA y MPS. No modifica la configuración del
    experimento ni retorna ningún valor.
    """
    torch_module = imported.get("PyTorch")
    if torch_module is None:
        return
    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = bool(
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    )
    print(f"\nDispositivos: CPU disponible, CUDA={cuda_available}, MPS={mps_available}")


def check_directories(root_dir: Path) -> list[str]:
    """
    ***
    * root_dir: Raíz del repositorio.
    ***
    Comprueba las carpetas mínimas y retorna sus rutas relativas cuando están ausentes.
    """
    missing = [path for path in REQUIRED_DIRECTORIES if not (root_dir / path).is_dir()]
    print("\nCarpetas")
    print("--------")
    for path in REQUIRED_DIRECTORIES:
        print(f"[{'ERROR' if path in missing else ' OK '}] {path}/")
    return missing


def main() -> int:
    """
    Ejecuta las comprobaciones de dependencias, dispositivos y carpetas.

    Retorna cero cuando el entorno está completo y uno si falta algún requisito.
    """
    argparse.ArgumentParser(
        description="Verifica dependencias y carpetas mínimas del experimento INF402."
    ).parse_args()
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
