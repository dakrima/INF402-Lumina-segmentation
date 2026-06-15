"""Placeholder for the future TIAToolbox BCSS baseline.

The target model is `fcn_resnet50_unet-bcss`. This module intentionally does
not download weights or run heavy inference yet. It records the intended entry
point so the pipeline can be wired in after environment and model availability
are verified.
"""

TARGET_MODEL_NAME = "fcn_resnet50_unet-bcss"


def describe_baseline() -> str:
    """Return a short description of the planned baseline."""
    return (
        f"Planned TIAToolbox baseline: {TARGET_MODEL_NAME}. "
        "Weights and real WSI inference are pending operational validation."
    )
