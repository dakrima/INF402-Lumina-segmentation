"""Patch selection architecture for INF402 Lumina/Histora."""

from src.selection.tiatoolbox_baseline import (
    BASELINE_SELECTOR_NAME,
    BaselineSelectionConfig,
    run_baseline_selection,
)

__all__ = [
    "BASELINE_SELECTOR_NAME",
    "BaselineSelectionConfig",
    "run_baseline_selection",
]
