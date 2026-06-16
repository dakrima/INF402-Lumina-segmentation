"""Patch selection architecture for INF402 Lumina/Histora."""

from src.selection.tiatoolbox_baseline import (
    BASELINE_SELECTOR_NAME,
    BaselineSelectionConfig,
    run_baseline_selection,
)
from src.selection.smart_tissue_nuclei import (
    SMART_SELECTOR_NAME,
    SmartTissueNucleiConfig,
    run_smart_tissue_nuclei_selection,
)

__all__ = [
    "BASELINE_SELECTOR_NAME",
    "BaselineSelectionConfig",
    "SMART_SELECTOR_NAME",
    "SmartTissueNucleiConfig",
    "run_baseline_selection",
    "run_smart_tissue_nuclei_selection",
]
