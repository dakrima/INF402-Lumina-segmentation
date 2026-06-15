#!/usr/bin/env python
"""Check the local project environment without downloading data or weights."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "PIL"),
    ("matplotlib", "matplotlib"),
    ("skimage", "skimage"),
    ("openslide", "openslide"),
    ("tiatoolbox", "tiatoolbox"),
    ("torch", "torch"),
]

REQUIRED_DIRECTORIES = [
    "data",
    "outputs",
    "src",
    "scripts",
]


def _module_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unavailable"))


def check_imports() -> tuple[list[str], dict[str, object]]:
    """Try all required imports and return failed package names."""
    failed: list[str] = []
    imported: dict[str, object] = {}

    print("\nPackage imports")
    print("---------------")
    for display_name, import_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            failed.append(display_name)
            print(f"[FAIL] {display_name}: {exc}")
            continue

        imported[display_name] = module
        print(f"[ OK ] {display_name}: {_module_version(module)}")

    return failed, imported


def check_cuda(imported: dict[str, object]) -> list[str]:
    """Report PyTorch CUDA status when torch is available."""
    warnings: list[str] = []
    torch_module = imported.get("torch")

    print("\nCUDA")
    print("----")
    if torch_module is None:
        print("[WARN] torch is not available; CUDA could not be checked.")
        warnings.append("CUDA check skipped because torch is missing.")
        return warnings

    cuda_available = bool(torch_module.cuda.is_available())
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        device_count = torch_module.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for device_index in range(device_count):
            print(f"GPU {device_index}: {torch_module.cuda.get_device_name(device_index)}")
    else:
        warnings.append("CUDA is not available in this environment.")
        print("[WARN] CUDA is not available. This is expected on many local CPU setups.")

    return warnings


def check_directories(root_dir: Path) -> list[str]:
    """Check that the expected project directories exist."""
    missing: list[str] = []

    print("\nProject directories")
    print("-------------------")
    for relative_path in REQUIRED_DIRECTORIES:
        directory = root_dir / relative_path
        if directory.is_dir():
            print(f"[ OK ] {relative_path}/")
        else:
            print(f"[FAIL] {relative_path}/ is missing")
            missing.append(relative_path)

    return missing


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]

    print("INF402 Lumina environment check")
    print("===============================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {root_dir}")

    failed_imports, imported = check_imports()
    cuda_warnings = check_cuda(imported)
    missing_directories = check_directories(root_dir)

    print("\nSummary")
    print("-------")
    if missing_directories:
        print("Environment incomplete: required project directories are missing.")
        print("Missing directories: " + ", ".join(missing_directories))
        return 1

    if failed_imports:
        print("Environment incomplete: required Python packages are missing or broken.")
        print("Failed imports: " + ", ".join(failed_imports))
        return 1

    if cuda_warnings:
        print("Environment OK with warnings.")
        for warning in cuda_warnings:
            print(f"- {warning}")
        return 0

    print("Environment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
