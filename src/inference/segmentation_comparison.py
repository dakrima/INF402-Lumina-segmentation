"""Compare technical segmentation runs produced from selected patches."""

from __future__ import annotations

import ast
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLINICAL_WARNING = (
    "Technical segmentation comparison only. Not for diagnosis, not RCB, "
    "not clinical validation."
)
REQUIRED_SEGMENTATION_FILES = [
    "per_patch_segmentation.csv",
    "inference_summary.json",
    "method_config.json",
]
REQUIRED_SEGMENTATION_DIRS = [
    "per_patch",
    "masks",
    "overlays",
    "overlays_with_legend",
    "input_previews",
]
METRICS_FIELDS = [
    "metric",
    "baseline_value",
    "smart_value",
    "delta_smart_minus_baseline",
    "relative_delta",
    "higher_is_better",
    "interpretation",
]
CLASS_DISTRIBUTION_FIELDS = [
    "class_id_or_name",
    "baseline_pixel_count",
    "smart_pixel_count",
    "baseline_ratio",
    "smart_ratio",
    "delta_smart_minus_baseline",
    "relative_delta",
]
PATCH_ROWS_FIELDS = [
    "method",
    "patch_id",
    "filename",
    "rank",
    "x_level0",
    "y_level0",
    "status",
    "raw_prediction_shape",
    "visualized_mask_shape",
    "resized_for_visualization",
    "num_patch_warnings",
    "patch_warnings",
    "unique_prediction_values",
    "class_pixel_counts",
    "dominant_class",
    "dominant_class_pixel_count",
    "dominant_class_ratio",
    "num_predicted_classes",
    "total_prediction_pixels",
    "mask_path",
    "overlay_path",
    "overlay_with_legend_path",
    "input_preview_path",
]
PREVIEW_SOURCE_TO_FIELD = {
    "overlays_with_legend": "overlay_with_legend_path",
    "overlays": "overlay_path",
    "masks": "mask_path",
    "input_previews": "input_preview_path",
}


@dataclass(frozen=True)
class SegmentationComparisonConfig:
    """Configuration for comparing two selected-patch segmentation runs."""

    baseline_seg_dir: Path
    smart_seg_dir: Path
    output_dir: Path
    root_dir: Path
    max_preview_patches: int = 8
    preview_source: str = "overlays_with_legend"
    overwrite: bool = False


@dataclass(frozen=True)
class RunData:
    """Loaded data for one segmentation run."""

    label: str
    seg_dir: Path
    summary: dict[str, Any]
    method_config: dict[str, Any]
    rows: list[dict[str, str]]
    method_name: str


def _resolve_path(path: Path, root_dir: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (root_dir / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _has_user_outputs(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(child.name != ".gitkeep" for child in output_dir.iterdir())


def _prepare_output_dir(output_dir: Path, root_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = root_dir.resolve()
    if _has_user_outputs(resolved_output) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {resolved_output}. "
            "Use --overwrite to regenerate this comparison."
        )

    if overwrite and resolved_output.exists():
        dangerous_paths = {
            Path("/").resolve(),
            Path.home().resolve(),
            resolved_root,
            resolved_root / "data",
            resolved_root / "outputs",
        }
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite only clears output directories inside the repository.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
        shutil.rmtree(resolved_output)

    output_dir.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pillow_modules() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required to generate segmentation comparison previews. "
            "Install project dependencies or run inside the inf402-lumina-seg environment."
        ) from exc
    return Image, ImageDraw, ImageFont


def _validate_segmentation_dir(seg_dir: Path, label: str) -> None:
    if not seg_dir.exists() or not seg_dir.is_dir():
        raise FileNotFoundError(f"{label} segmentation directory does not exist: {seg_dir}")
    for file_name in REQUIRED_SEGMENTATION_FILES:
        path = seg_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"{label} segmentation run is missing required file: {path}")
    for dir_name in REQUIRED_SEGMENTATION_DIRS:
        path = seg_dir / dir_name
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"{label} segmentation run is missing required directory: {path}")


def _json_cell(value: object) -> str:
    if value in ("", None):
        return ""
    return json.dumps(value, sort_keys=True)


def _parse_structured_cell(value: str | None) -> Any:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def _parse_mapping_cell(value: str | None) -> dict[str, int]:
    parsed = _parse_structured_cell(value)
    if not isinstance(parsed, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in parsed.items():
        try:
            counts[str(key)] = int(count)
        except (TypeError, ValueError):
            continue
    return counts


def _parse_sequence_cell(value: str | None) -> list[Any]:
    parsed = _parse_structured_cell(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return []


def _parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return default


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _relative_delta(baseline: float | int | None, smart: float | int | None) -> str:
    if baseline in (None, 0) or smart is None:
        return ""
    return _format_number((float(smart) - float(baseline)) / float(baseline))


def _numeric_delta(baseline: float | int | None, smart: float | int | None) -> str:
    if baseline is None or smart is None:
        return ""
    return _format_number(float(smart) - float(baseline))


def _class_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _dominant_class(counts: dict[str, int]) -> tuple[str, int, float]:
    positive = {key: count for key, count in counts.items() if count > 0}
    total = sum(positive.values())
    if not positive or total <= 0:
        return "", 0, 0.0
    dominant_key, dominant_count = max(
        positive.items(),
        key=lambda item: (item[1], _class_sort_key(item[0])),
    )
    return dominant_key, dominant_count, dominant_count / total


def _shape_product(shape: list[Any]) -> int:
    if len(shape) < 2:
        return 0
    try:
        return int(shape[0]) * int(shape[1])
    except (TypeError, ValueError):
        return 0


def _load_run(seg_dir: Path, label: str) -> RunData:
    _validate_segmentation_dir(seg_dir, label)
    summary = _read_json(seg_dir / "inference_summary.json")
    method_config = _read_json(seg_dir / "method_config.json")
    rows = _read_csv(seg_dir / "per_patch_segmentation.csv")
    method_name = (
        str(summary.get("selection_method") or "")
        or str(summary.get("selector") or "")
        or label
    )
    return RunData(
        label=label,
        seg_dir=seg_dir,
        summary=summary,
        method_config=method_config,
        rows=rows,
        method_name=method_name,
    )


def _completed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "completed"]


def _dominant_shape(rows: list[dict[str, str]], field: str) -> list[Any]:
    counter: Counter[str] = Counter()
    parsed_by_key: dict[str, list[Any]] = {}
    for row in _completed_rows(rows):
        shape = _parse_sequence_cell(row.get(field))
        if not shape:
            continue
        key = json.dumps(shape, sort_keys=True)
        counter[key] += 1
        parsed_by_key[key] = shape
    if not counter:
        return []
    return parsed_by_key[counter.most_common(1)[0][0]]


def _row_class_counts(rows: list[dict[str, str]]) -> list[dict[str, int]]:
    return [_parse_mapping_cell(row.get("class_pixel_counts")) for row in _completed_rows(rows)]


def _run_class_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counts in _row_class_counts(rows):
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _run_numeric_summary(run: RunData) -> dict[str, float | int | None]:
    completed_rows = _completed_rows(run.rows)
    class_counts = _row_class_counts(run.rows)
    class_distribution = _run_class_distribution(run.rows)
    dominant_ratios: list[float] = []
    predicted_class_counts: list[int] = []
    prediction_areas: list[int] = []
    visualized_areas: list[int] = []

    for row, counts in zip(completed_rows, class_counts, strict=False):
        _, _, dominant_ratio = _dominant_class(counts)
        if counts:
            dominant_ratios.append(dominant_ratio)
            predicted_class_counts.append(sum(1 for count in counts.values() if count > 0))
            prediction_areas.append(sum(counts.values()))
        visualized_shape = _parse_sequence_cell(row.get("visualized_mask_shape"))
        visualized_area = _shape_product(visualized_shape)
        if visualized_area:
            visualized_areas.append(visualized_area)

    completed = int(run.summary.get("num_patches_completed", len(completed_rows)) or 0)
    runtime = _parse_float(run.summary.get("runtime_seconds"), 0.0)
    return {
        "num_completed_patches": completed,
        "num_failed_patches": int(run.summary.get("num_patches_failed", 0) or 0),
        "num_skipped_patches": int(run.summary.get("num_patches_skipped", 0) or 0),
        "runtime_seconds": runtime,
        "mean_runtime_per_completed_patch": runtime / completed if completed else None,
        "num_patch_warnings": int(run.summary.get("num_patch_warnings", 0) or 0),
        "num_patches_with_resized_visualization": int(
            run.summary.get("num_patches_with_resized_visualization", 0) or 0
        ),
        "num_unique_predicted_classes": len(class_distribution),
        "mean_num_predicted_classes_per_patch": (
            sum(predicted_class_counts) / len(predicted_class_counts)
            if predicted_class_counts
            else None
        ),
        "mean_dominant_class_ratio": (
            sum(dominant_ratios) / len(dominant_ratios) if dominant_ratios else None
        ),
        "mean_non_background_ratio": None,
        "mean_prediction_area_pixels": (
            sum(prediction_areas) / len(prediction_areas) if prediction_areas else None
        ),
        "mean_visualized_area_pixels": (
            sum(visualized_areas) / len(visualized_areas) if visualized_areas else None
        ),
    }


def _metric_rows(
    baseline: RunData,
    smart: RunData,
    validation_warnings: list[str],
) -> list[dict[str, object]]:
    baseline_values = _run_numeric_summary(baseline)
    smart_values = _run_numeric_summary(smart)
    validation_warnings.append(
        "mean_non_background_ratio was not computed because the model output does not "
        "define a safe background class for this technical comparison."
    )
    metric_specs = [
        ("num_completed_patches", "true", "Completed technical inference rows."),
        ("num_failed_patches", "false", "Lower is better operationally."),
        ("num_skipped_patches", "false", "Lower is better operationally."),
        ("runtime_seconds", "false", "Wall-clock runtime reported by each segmentation run."),
        (
            "mean_runtime_per_completed_patch",
            "false",
            "Runtime divided by completed patches.",
        ),
        ("num_patch_warnings", "false", "Patch-level technical warnings from inference."),
        (
            "num_patches_with_resized_visualization",
            "false",
            "Patches where raw mask resolution differed from visualization resolution.",
        ),
        (
            "num_unique_predicted_classes",
            "context_dependent",
            "Unique class IDs predicted across completed patches; not clinical truth.",
        ),
        (
            "mean_num_predicted_classes_per_patch",
            "context_dependent",
            "Average class variety per completed patch.",
        ),
        (
            "mean_dominant_class_ratio",
            "false",
            "Lower values indicate less dominance by a single predicted class.",
        ),
        (
            "mean_non_background_ratio",
            "",
            "Not computed without an explicit background class definition.",
        ),
        (
            "mean_prediction_area_pixels",
            "context_dependent",
            "Average raw prediction area in pixels.",
        ),
        (
            "mean_visualized_area_pixels",
            "context_dependent",
            "Average visualization mask area in pixels.",
        ),
    ]
    rows: list[dict[str, object]] = []
    for metric, higher_is_better, interpretation in metric_specs:
        baseline_value = baseline_values.get(metric)
        smart_value = smart_values.get(metric)
        rows.append(
            {
                "metric": metric,
                "baseline_value": _format_number(baseline_value),
                "smart_value": _format_number(smart_value),
                "delta_smart_minus_baseline": _numeric_delta(baseline_value, smart_value),
                "relative_delta": _relative_delta(baseline_value, smart_value),
                "higher_is_better": higher_is_better,
                "interpretation": interpretation,
            }
        )
    return rows


def _class_distribution_rows(
    baseline: RunData,
    smart: RunData,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float | int]]]:
    baseline_distribution = _run_class_distribution(baseline.rows)
    smart_distribution = _run_class_distribution(smart.rows)
    baseline_total = sum(baseline_distribution.values())
    smart_total = sum(smart_distribution.values())
    class_ids = sorted(
        set(baseline_distribution) | set(smart_distribution),
        key=_class_sort_key,
    )
    rows: list[dict[str, object]] = []
    summary_distribution: dict[str, dict[str, float | int]] = {}
    for class_id in class_ids:
        baseline_count = baseline_distribution.get(class_id, 0)
        smart_count = smart_distribution.get(class_id, 0)
        baseline_ratio = baseline_count / baseline_total if baseline_total else 0.0
        smart_ratio = smart_count / smart_total if smart_total else 0.0
        rows.append(
            {
                "class_id_or_name": class_id,
                "baseline_pixel_count": baseline_count,
                "smart_pixel_count": smart_count,
                "baseline_ratio": _format_number(baseline_ratio),
                "smart_ratio": _format_number(smart_ratio),
                "delta_smart_minus_baseline": smart_count - baseline_count,
                "relative_delta": _relative_delta(baseline_count, smart_count),
            }
        )
        summary_distribution[class_id] = {
            "baseline_pixel_count": baseline_count,
            "smart_pixel_count": smart_count,
            "baseline_ratio": baseline_ratio,
            "smart_ratio": smart_ratio,
            "delta_smart_minus_baseline": smart_count - baseline_count,
        }
    return rows, summary_distribution


def _patch_rows_for_run(run: RunData) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in run.rows:
        counts = _parse_mapping_cell(row.get("class_pixel_counts"))
        dominant_class, dominant_count, dominant_ratio = _dominant_class(counts)
        total_prediction_pixels = sum(counts.values())
        unique_prediction_values = _parse_sequence_cell(row.get("unique_prediction_values"))
        raw_prediction_shape = _parse_sequence_cell(
            row.get("raw_prediction_shape") or row.get("prediction_shape")
        )
        visualized_mask_shape = _parse_sequence_cell(row.get("visualized_mask_shape"))
        patch_warnings = _parse_sequence_cell(row.get("patch_warnings"))
        output_rows.append(
            {
                "method": run.method_name,
                "patch_id": row.get("patch_id", ""),
                "filename": row.get("filename", ""),
                "rank": row.get("rank", ""),
                "x_level0": row.get("x_level0", ""),
                "y_level0": row.get("y_level0", ""),
                "status": row.get("status", ""),
                "raw_prediction_shape": _json_cell(raw_prediction_shape),
                "visualized_mask_shape": _json_cell(visualized_mask_shape),
                "resized_for_visualization": str(_parse_bool(row.get("resized_for_visualization"))).lower(),
                "num_patch_warnings": _parse_int(row.get("num_patch_warnings")),
                "patch_warnings": _json_cell(patch_warnings),
                "unique_prediction_values": _json_cell(unique_prediction_values),
                "class_pixel_counts": _json_cell(counts),
                "dominant_class": dominant_class,
                "dominant_class_pixel_count": dominant_count,
                "dominant_class_ratio": _format_number(dominant_ratio),
                "num_predicted_classes": sum(1 for count in counts.values() if count > 0),
                "total_prediction_pixels": total_prediction_pixels,
                "mask_path": row.get("mask_path", ""),
                "overlay_path": row.get("overlay_path", ""),
                "overlay_with_legend_path": row.get("overlay_with_legend_path", ""),
                "input_preview_path": row.get("input_preview_path", ""),
            }
        )
    return output_rows


def _validation_warnings(baseline: RunData, smart: RunData) -> list[str]:
    warnings: list[str] = []
    if baseline.summary.get("model_name") != smart.summary.get("model_name"):
        warnings.append(
            "Model name differs between segmentation runs: "
            f"{baseline.summary.get('model_name')} vs {smart.summary.get('model_name')}."
        )
    if baseline.summary.get("input_mode") != smart.summary.get("input_mode"):
        warnings.append(
            "Input mode differs between segmentation runs: "
            f"{baseline.summary.get('input_mode')} vs {smart.summary.get('input_mode')}."
        )
    if baseline.summary.get("overlay_alpha") != smart.summary.get("overlay_alpha"):
        warnings.append(
            "Overlay alpha differs between segmentation runs: "
            f"{baseline.summary.get('overlay_alpha')} vs {smart.summary.get('overlay_alpha')}."
        )
    if baseline.summary.get("num_patches_completed") != smart.summary.get("num_patches_completed"):
        warnings.append(
            "Completed patch counts differ between segmentation runs: "
            f"{baseline.summary.get('num_patches_completed')} vs "
            f"{smart.summary.get('num_patches_completed')}."
        )
    for run in [baseline, smart]:
        failed = int(run.summary.get("num_patches_failed", 0) or 0)
        skipped = int(run.summary.get("num_patches_skipped", 0) or 0)
        patch_warnings = int(run.summary.get("num_patch_warnings", 0) or 0)
        resized = int(run.summary.get("num_patches_with_resized_visualization", 0) or 0)
        if failed:
            warnings.append(f"{run.method_name} has {failed} failed patch inferences.")
        if skipped:
            warnings.append(f"{run.method_name} has {skipped} skipped patch rows.")
        if patch_warnings:
            warnings.append(f"{run.method_name} has {patch_warnings} patch-level warnings.")
        if resized:
            warnings.append(
                f"{run.method_name} has {resized} patches with resized visualization masks."
            )
    baseline_raw_shape = _dominant_shape(baseline.rows, "raw_prediction_shape")
    smart_raw_shape = _dominant_shape(smart.rows, "raw_prediction_shape")
    if baseline_raw_shape != smart_raw_shape:
        warnings.append(
            "Dominant raw prediction resolution differs: "
            f"{baseline_raw_shape} vs {smart_raw_shape}."
        )
    baseline_visual_shape = _dominant_shape(baseline.rows, "visualized_mask_shape")
    smart_visual_shape = _dominant_shape(smart.rows, "visualized_mask_shape")
    if baseline_visual_shape != smart_visual_shape:
        warnings.append(
            "Dominant visualized mask resolution differs: "
            f"{baseline_visual_shape} vs {smart_visual_shape}."
        )
    return warnings


def _open_preview_image(path: str, size: tuple[int, int]) -> Image.Image:
    Image, _, _ = _pillow_modules()
    if not path:
        return _placeholder_image(size, "missing path")
    image_path = Path(path)
    if not image_path.exists():
        return _placeholder_image(size, "missing file")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        return _placeholder_image(size, f"load failed: {exc}")


def _placeholder_image(size: tuple[int, int], text: str) -> Image.Image:
    Image, ImageDraw, ImageFont = _pillow_modules()
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(180, 180, 180))
    draw.multiline_text((12, 12), text[:120], fill=(80, 80, 80), font=font, spacing=4)
    return image


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    Image, _, _ = _pillow_modules()
    fitted = image.copy()
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    fitted.thumbnail(size, resampling)
    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_chars: int,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:max_chars]
    if current:
        lines.append(current)
    draw.multiline_text(xy, "\n".join(lines[:2]), fill=fill, font=font, spacing=2)


def _create_preview(
    baseline: RunData,
    smart: RunData,
    output_path: Path,
    *,
    max_preview_patches: int,
    preview_source: str,
) -> Path:
    Image, ImageDraw, ImageFont = _pillow_modules()
    field = PREVIEW_SOURCE_TO_FIELD[preview_source]
    baseline_rows = _completed_rows(baseline.rows)[:max_preview_patches]
    smart_rows = _completed_rows(smart.rows)[:max_preview_patches]
    row_count = max(len(baseline_rows), len(smart_rows), 1)
    image_size = (360, 360)
    title_height = 78
    row_gap = 20
    margin = 24
    column_gap = 24
    column_width = image_size[0]
    width = margin * 2 + column_width * 2 + column_gap
    height = margin * 2 + 44 + row_count * (title_height + image_size[1] + row_gap)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header_font = font
    draw.text(
        (margin, margin),
        f"Technical segmentation comparison - preview: {preview_source}",
        fill=(25, 25, 25),
        font=header_font,
    )
    draw.text((margin, margin + 22), baseline.method_name, fill=(25, 25, 25), font=font)
    smart_x = margin + column_width + column_gap
    draw.text((smart_x, margin + 22), smart.method_name, fill=(25, 25, 25), font=font)

    start_y = margin + 52
    for index in range(row_count):
        row_y = start_y + index * (title_height + image_size[1] + row_gap)
        for x, rows, empty_label in [
            (margin, baseline_rows, "no baseline image"),
            (smart_x, smart_rows, "no smart image"),
        ]:
            if index < len(rows):
                row = rows[index]
                title = f"{row.get('patch_id', '')} | rank {row.get('rank', '')}".strip()
                preview_image = _open_preview_image(row.get(field, ""), image_size)
            else:
                title = empty_label
                preview_image = _placeholder_image(image_size, empty_label)
            _draw_wrapped_text(
                draw,
                title,
                (x, row_y),
                font=font,
                fill=(35, 35, 35),
                max_chars=48,
            )
            canvas.paste(_fit_image(preview_image, image_size), (x, row_y + title_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _markdown_table(rows: list[dict[str, object]], columns: list[str], max_rows: int | None = None) -> str:
    selected_rows = rows[:max_rows] if max_rows is not None else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected_rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_notes(
    path: Path,
    *,
    baseline: RunData,
    smart: RunData,
    metrics_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    summary: dict[str, Any],
) -> Path:
    lines = [
        "# Comparacion tecnica de segmentacion sobre patches seleccionados",
        "",
        f"Fecha UTC: {_utc_now_iso()}",
        "",
        "## Objetivo",
        "",
        (
            "Comparar outputs tecnicos de segmentacion generados sobre patches "
            "seleccionados por dos metodos. Esta comparacion no usa ground truth y "
            "no debe interpretarse como diagnostico ni validacion clinica."
        ),
        "",
        "## Entradas",
        "",
        f"- Baseline: `{baseline.seg_dir}`",
        f"- Smart: `{smart.seg_dir}`",
        f"- Warning clinico: {CLINICAL_WARNING}",
        "",
        "## Resumen",
        "",
        f"- Estado: `{summary['status']}`",
        f"- Baseline completados/fallidos/saltados: {summary['num_baseline_completed']}/"
        f"{summary['num_baseline_failed']}/{summary['num_baseline_skipped']}",
        f"- Smart completados/fallidos/saltados: {summary['num_smart_completed']}/"
        f"{summary['num_smart_failed']}/{summary['num_smart_skipped']}",
        f"- Warnings baseline/smart: {summary['baseline_num_patch_warnings']}/"
        f"{summary['smart_num_patch_warnings']}",
        f"- Visualizaciones reescaladas baseline/smart: "
        f"{summary['baseline_num_resized_visualizations']}/"
        f"{summary['smart_num_resized_visualizations']}",
        "",
        "## Metricas",
        "",
        _markdown_table(
            metrics_rows,
            [
                "metric",
                "baseline_value",
                "smart_value",
                "delta_smart_minus_baseline",
                "interpretation",
            ],
        ),
        "",
        "## Distribucion de clases predichas",
        "",
        _markdown_table(
            class_rows,
            [
                "class_id_or_name",
                "baseline_pixel_count",
                "smart_pixel_count",
                "baseline_ratio",
                "smart_ratio",
                "delta_smart_minus_baseline",
            ],
        ),
        "",
        "## Nota sobre resolucion",
        "",
        (
            "La mascara cruda puede tener resolucion 512x512 mientras que el overlay "
            "visual puede tener resolucion 1024x1024. El reescalado con vecino mas "
            "cercano preserva etiquetas discretas. Los conteos `class_pixel_counts` "
            "corresponden a la resolucion cruda de prediccion, no al overlay."
        ),
        "",
        "## Limitaciones",
        "",
        "- Comparacion tecnica sobre patches seleccionados.",
        "- No usa ground truth.",
        "- No diagnostica.",
        "- No calcula RCB.",
        "- No constituye validacion clinica.",
        "- No permite afirmar superioridad clinica a partir de estas metricas.",
    ]
    if summary["validation_warnings"]:
        lines.extend(
            [
                "",
                "## Warnings de validacion",
                "",
                *[f"- {warning}" for warning in summary["validation_warnings"]],
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def compare_segmentation_runs(config: SegmentationComparisonConfig) -> dict[str, Any]:
    """Compare two technical segmentation runs and write summary artifacts."""
    root_dir = config.root_dir.resolve()
    baseline_seg_dir = _resolve_path(config.baseline_seg_dir, root_dir)
    smart_seg_dir = _resolve_path(config.smart_seg_dir, root_dir)
    output_dir = _resolve_path(config.output_dir, root_dir)
    if config.max_preview_patches <= 0:
        raise ValueError("--max-preview-patches must be positive.")
    if config.preview_source not in PREVIEW_SOURCE_TO_FIELD:
        raise ValueError(
            "--preview-source must be one of: "
            + ", ".join(sorted(PREVIEW_SOURCE_TO_FIELD))
        )

    baseline = _load_run(baseline_seg_dir, "baseline")
    smart = _load_run(smart_seg_dir, "smart")
    _prepare_output_dir(output_dir=output_dir, root_dir=root_dir, overwrite=config.overwrite)

    validation_warnings = _validation_warnings(baseline, smart)
    metrics_rows = _metric_rows(baseline, smart, validation_warnings)
    class_rows, class_distribution = _class_distribution_rows(baseline, smart)
    patch_rows = _patch_rows_for_run(baseline) + _patch_rows_for_run(smart)

    metrics_path = output_dir / "segmentation_comparison_metrics.csv"
    class_distribution_path = output_dir / "segmentation_class_distribution.csv"
    patch_rows_path = output_dir / "segmentation_patch_rows.csv"
    preview_path = output_dir / "segmentation_comparison_preview.png"
    notes_path = output_dir / "segmentation_comparison_notes.md"
    summary_path = output_dir / "segmentation_comparison_summary.json"

    _write_csv(metrics_rows, metrics_path, METRICS_FIELDS)
    _write_csv(class_rows, class_distribution_path, CLASS_DISTRIBUTION_FIELDS)
    _write_csv(patch_rows, patch_rows_path, PATCH_ROWS_FIELDS)
    _create_preview(
        baseline,
        smart,
        preview_path,
        max_preview_patches=config.max_preview_patches,
        preview_source=config.preview_source,
    )

    status = "completed_with_warnings" if validation_warnings else "completed"
    summary = {
        "status": status,
        "baseline_seg_dir": str(baseline_seg_dir),
        "smart_seg_dir": str(smart_seg_dir),
        "output_dir": str(output_dir),
        "baseline_selector": baseline.summary.get("selector") or baseline.method_name,
        "smart_selector": smart.summary.get("selector") or smart.method_name,
        "model_name_baseline": baseline.summary.get("model_name", ""),
        "model_name_smart": smart.summary.get("model_name", ""),
        "input_mode_baseline": baseline.summary.get("input_mode", ""),
        "input_mode_smart": smart.summary.get("input_mode", ""),
        "num_baseline_rows": len(baseline.rows),
        "num_smart_rows": len(smart.rows),
        "num_baseline_completed": int(baseline.summary.get("num_patches_completed", 0) or 0),
        "num_smart_completed": int(smart.summary.get("num_patches_completed", 0) or 0),
        "num_baseline_failed": int(baseline.summary.get("num_patches_failed", 0) or 0),
        "num_smart_failed": int(smart.summary.get("num_patches_failed", 0) or 0),
        "num_baseline_skipped": int(baseline.summary.get("num_patches_skipped", 0) or 0),
        "num_smart_skipped": int(smart.summary.get("num_patches_skipped", 0) or 0),
        "baseline_runtime_seconds": _parse_float(baseline.summary.get("runtime_seconds")),
        "smart_runtime_seconds": _parse_float(smart.summary.get("runtime_seconds")),
        "baseline_num_patch_warnings": int(baseline.summary.get("num_patch_warnings", 0) or 0),
        "smart_num_patch_warnings": int(smart.summary.get("num_patch_warnings", 0) or 0),
        "baseline_num_resized_visualizations": int(
            baseline.summary.get("num_patches_with_resized_visualization", 0) or 0
        ),
        "smart_num_resized_visualizations": int(
            smart.summary.get("num_patches_with_resized_visualization", 0) or 0
        ),
        "validation_warnings": validation_warnings,
        "class_distribution": class_distribution,
        "outputs": {
            "segmentation_comparison_summary_json": str(summary_path),
            "segmentation_comparison_metrics_csv": str(metrics_path),
            "segmentation_class_distribution_csv": str(class_distribution_path),
            "segmentation_patch_rows_csv": str(patch_rows_path),
            "segmentation_comparison_preview_png": str(preview_path),
            "segmentation_comparison_notes_md": str(notes_path),
        },
        "clinical_warning": CLINICAL_WARNING,
    }
    _write_notes(
        notes_path,
        baseline=baseline,
        smart=smart,
        metrics_rows=metrics_rows,
        class_rows=class_rows,
        summary=summary,
    )
    _write_json(summary, summary_path)
    return summary
