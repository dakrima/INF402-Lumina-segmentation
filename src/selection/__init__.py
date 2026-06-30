"""Selectores que participan en el experimento final de INF402."""

# Primary paper selectors.
from src.selection.tiatoolbox_baseline import (
    BASELINE_SELECTOR_NAME,
    BaselineSelectionConfig,
    run_baseline_selection,
)
from src.selection.proposed_selector import (
    V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME,
    V41MedicalEmbeddingAssistedConfig,
    run_v4_1_medical_embedding_assisted_selection,
)

__all__ = [
    "BASELINE_SELECTOR_NAME",
    "BaselineSelectionConfig",
    "V41_MEDICAL_EMBEDDING_ASSISTED_SELECTOR_NAME",
    "V41MedicalEmbeddingAssistedConfig",
    "run_baseline_selection",
    "run_v4_1_medical_embedding_assisted_selection",
]
