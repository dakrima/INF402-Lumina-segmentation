"""Centralized filesystem paths for the project."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
PATCHES_DIR = OUTPUTS_DIR / "patches"
MASKS_DIR = OUTPUTS_DIR / "masks"
OVERLAYS_DIR = OUTPUTS_DIR / "overlays"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_directories() -> None:
    """Create the standard data and output directories if they are missing."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        EXTERNAL_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        PATCHES_DIR,
        MASKS_DIR,
        OVERLAYS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
