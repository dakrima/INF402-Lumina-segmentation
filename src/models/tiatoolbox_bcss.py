"""Smoke-test helpers for the TIAToolbox BCSS pretrained model."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "fcn_resnet50_unet-bcss"
DEFAULT_STATUS_JSON = Path("outputs/model_checks/tiatoolbox_bcss_model_status.json")
SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}


def _safe_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unavailable"))


def _import_optional_module(module_name: str) -> tuple[object | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001 - smoke-test diagnostic
        return None, str(exc)


def _check_tiatoolbox_home(tiatoolbox_module: object, root_dir: Path) -> str | None:
    """Return a safety error if TIAToolbox cache points inside the repository."""
    rc_param = getattr(tiatoolbox_module, "rcParam", None)
    if not isinstance(rc_param, dict):
        return None

    tiatoolbox_home = rc_param.get("TIATOOLBOX_HOME")
    if tiatoolbox_home is None:
        return None

    home_path = Path(tiatoolbox_home).expanduser().resolve()
    repo_root = root_dir.resolve()
    try:
        home_path.relative_to(repo_root)
    except ValueError:
        return None

    return (
        "TIAToolbox cache directory points inside the repository "
        f"({home_path}). Move TIATOOLBOX_HOME outside the repo before loading weights."
    )


def resolve_torch_device(torch_module: object, requested_device: str) -> str:
    """Resolve a requested device using safe PyTorch availability checks."""
    if requested_device not in SUPPORTED_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"Unsupported device '{requested_device}'. Choose one of: {supported}")

    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = bool(
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )

    if requested_device == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if requested_device == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")

    return requested_device


def _move_model_to_device(
    model: object,
    resolved_device: str,
    requested_device: str,
) -> tuple[str, list[str]]:
    """Move a model to the requested device, falling back from MPS when needed."""
    warnings: list[str] = []

    if resolved_device == "mps":
        try:
            model.to("mps")
            return "mps", warnings
        except Exception as exc:  # noqa: BLE001 - smoke-test diagnostic
            if requested_device == "auto":
                warnings.append(
                    "Moving model to MPS failed; falling back to CPU. "
                    f"MPS error: {exc}"
                )
                model.to("cpu")
                return "cpu", warnings
            raise

    model.to(resolved_device)
    return resolved_device, warnings


def _empty_status(model_name: str, requested_device: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "status": "failed",
        "tiatoolbox_version": None,
        "torch_version": None,
        "requested_device": requested_device,
        "resolved_device": None,
        "cuda_available": None,
        "mps_available": None,
        "model_class": None,
        "ioconfig_class": None,
        "tiatoolbox_home": None,
        "error": None,
        "warnings": [],
        "suggested_next_step": None,
    }


def build_model_status(
    model_name: str = DEFAULT_MODEL_NAME,
    requested_device: str = "auto",
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """Attempt to load a TIAToolbox pretrained model and return status metadata."""
    root_dir = Path.cwd() if root_dir is None else root_dir
    status = _empty_status(model_name=model_name, requested_device=requested_device)

    torch_module, torch_error = _import_optional_module("torch")
    tiatoolbox_module, tiatoolbox_error = _import_optional_module("tiatoolbox")

    if torch_module is None:
        status["resolved_device"] = "cpu"
    else:
        status["torch_version"] = _safe_version(torch_module)
        status["cuda_available"] = bool(torch_module.cuda.is_available())
        status["mps_available"] = bool(
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )

    if tiatoolbox_module is not None:
        status["tiatoolbox_version"] = _safe_version(tiatoolbox_module)
        rc_param = getattr(tiatoolbox_module, "rcParam", {})
        if isinstance(rc_param, dict) and rc_param.get("TIATOOLBOX_HOME") is not None:
            status["tiatoolbox_home"] = str(Path(rc_param["TIATOOLBOX_HOME"]).expanduser())

    missing_dependencies: list[str] = []
    if torch_module is None:
        missing_dependencies.append(f"torch ({torch_error})")
    if tiatoolbox_module is None:
        missing_dependencies.append(f"tiatoolbox ({tiatoolbox_error})")
    if missing_dependencies:
        status["error"] = "Missing dependency: " + "; ".join(missing_dependencies)
        status["suggested_next_step"] = (
            "Activate the Conda/Mamba environment and install the missing dependencies."
        )
        return status

    try:
        resolved_device = resolve_torch_device(torch_module, requested_device)
    except Exception as exc:  # noqa: BLE001 - smoke-test diagnostic
        status["error"] = str(exc)
        status["resolved_device"] = "cpu"
        status["suggested_next_step"] = "Use --device auto or run on a machine with the requested backend."
        return status

    status["resolved_device"] = resolved_device

    cache_error = _check_tiatoolbox_home(tiatoolbox_module, root_dir=root_dir)
    if cache_error is not None:
        status["error"] = cache_error
        status["suggested_next_step"] = "Set TIATOOLBOX_HOME to a cache directory outside this repository."
        return status

    try:
        architecture_module = importlib.import_module("tiatoolbox.models.architecture")
        get_pretrained_model = getattr(architecture_module, "get_pretrained_model")
        loaded_model = get_pretrained_model(pretrained_model=model_name)
        if isinstance(loaded_model, tuple):
            model = loaded_model[0]
            ioconfig = loaded_model[1] if len(loaded_model) > 1 else None
        else:
            model = loaded_model
            ioconfig = None

        final_device, warnings = _move_model_to_device(
            model=model,
            resolved_device=resolved_device,
            requested_device=requested_device,
        )
        status["resolved_device"] = final_device
        status["warnings"].extend(warnings)

        if hasattr(model, "eval"):
            model.eval()

        status["status"] = "loaded"
        status["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
        if ioconfig is not None:
            status["ioconfig_class"] = (
                f"{ioconfig.__class__.__module__}.{ioconfig.__class__.__name__}"
            )
        status["error"] = None
        status["suggested_next_step"] = (
            "Proceed to a controlled inference smoke test on a small local image."
        )
        return status
    except Exception as exc:  # noqa: BLE001 - smoke-test diagnostic
        status["status"] = "failed"
        status["error"] = str(exc)
        status["suggested_next_step"] = (
            "Confirm TIAToolbox version, network/cache access for pretrained weights, "
            "and that the model name is available in this installation."
        )
        return status


def write_model_status_json(status: dict[str, Any], output_json: str | Path) -> Path:
    """Write a model status dictionary as pretty JSON."""
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
