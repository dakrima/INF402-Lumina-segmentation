"""Patch selection architecture for INF402 Lumina/Histora."""

from src.selection.tiatoolbox_baseline import (
    BASELINE_SELECTOR_NAME,
    BaselineSelectionConfig,
    run_baseline_selection,
)
from src.selection.smart_tissue_nuclei import (
    SMART_SELECTOR_NAME,
    SMART_V2_LIGHT_SELECTOR_NAME,
    SmartTissueNucleiConfig,
    run_smart_tissue_nuclei_selection,
)
from src.selection.v3_server_quality import (
    V3_SERVER_QUALITY_SELECTOR_NAME,
    V3ServerQualityConfig,
    run_v3_server_quality_selection,
)

__all__ = [
    "BASELINE_SELECTOR_NAME",
    "BaselineSelectionConfig",
    "SMART_SELECTOR_NAME",
    "SMART_V2_LIGHT_SELECTOR_NAME",
    "SmartTissueNucleiConfig",
    "V3_SERVER_QUALITY_SELECTOR_NAME",
    "V3ServerQualityConfig",
    "run_baseline_selection",
    "run_smart_tissue_nuclei_selection",
    "run_v3_server_quality_selection",
]
