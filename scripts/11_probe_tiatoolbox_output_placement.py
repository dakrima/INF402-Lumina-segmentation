#!/usr/bin/env python
"""Probe TIAToolbox 512x512 output placement for 1024x1024 patch inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.output_placement_probe import (  # noqa: E402
    CLINICAL_WARNING,
    determine_overall_conclusion,
    inspect_tiatoolbox_output_placement,
    run_coordinate_probe,
    run_offset_search,
    run_secondary_consistency_check,
    run_tiatoolbox_merge_probe,
    selected_patch_coordinates,
    selected_patch_path,
    write_json,
    write_output_placement_report_md,
)
from src.models.tiatoolbox_bcss import DEFAULT_MODEL_NAME, SUPPORTED_DEVICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Investigate where TIAToolbox places a 512x512 model output inside "
            "a 1024x1024 input patch. This is technical inference only; it does "
            "not diagnose, calculate RCB, or validate clinical performance."
        ),
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs"),
        help="Patch selection run containing selected_metadata.csv and selected PNGs.",
    )
    parser.add_argument("--patch-index", type=int, default=0, help="Zero-based selected patch row.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/context_stitch_probe/output_placement"),
        help="Output directory for placement probe reports.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="cpu",
        help="Device for optional direct inference consistency check.",
    )
    parser.add_argument("--run-source-inspection", action="store_true")
    parser.add_argument("--run-coordinate-probe", action="store_true")
    parser.add_argument("--run-tiatoolbox-merge-probe", action="store_true")
    parser.add_argument("--run-secondary-consistency-check", action="store_true")
    parser.add_argument("--run-offset-search", action="store_true")
    parser.add_argument(
        "--offset-search-step",
        type=int,
        default=16,
        help="Pixel step for optional secondary offset search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output directory. Only safe repo output paths are cleared.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (ROOT_DIR / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = ROOT_DIR.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        resolved_root,
        resolved_root / "data",
        resolved_root / "outputs",
        resolved_root / "outputs" / "context_stitch_probe",
    }
    if resolved_output.exists() and any(resolved_output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {resolved_output}. "
            "Use --overwrite to regenerate probe outputs."
        )
    if overwrite and resolved_output.exists():
        if not _is_relative_to(resolved_output, resolved_root):
            raise ValueError("--overwrite only clears output directories inside the repository.")
        if resolved_output in dangerous_paths:
            raise ValueError(f"Refusing to clear dangerous output path: {resolved_output}")
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def _write_failure(output_dir: Path, error: str) -> Path:
    payload = {
        "status": "failed",
        "error": error,
        "clinical_warning": CLINICAL_WARNING,
    }
    return write_json(payload, output_dir / "output_placement_summary.json")


def _flag_defaults(args: argparse.Namespace) -> None:
    any_probe_requested = any(
        (
            args.run_source_inspection,
            args.run_coordinate_probe,
            args.run_tiatoolbox_merge_probe,
            args.run_secondary_consistency_check,
            args.run_offset_search,
        )
    )
    if not any_probe_requested:
        args.run_source_inspection = True
        args.run_coordinate_probe = True


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    args = parse_args()
    _flag_defaults(args)

    selection_dir = _resolve_path(args.selection_dir)
    output_dir = _resolve_path(args.output_dir)

    print("TIAToolbox output placement probe")
    print("=================================")
    print(f"Selection dir: {selection_dir}")
    print(f"Patch index: {args.patch_index}")
    print(f"Output dir: {output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Requested device: {args.device}")
    print(f"Clinical warning: {CLINICAL_WARNING}")

    try:
        _prepare_output_dir(output_dir=output_dir, overwrite=args.overwrite)
        if not selection_dir.exists():
            raise FileNotFoundError(f"Selection directory does not exist: {selection_dir}")

        report: dict[str, Any] = {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_dir": str(selection_dir),
            "patch_index": args.patch_index,
            "output_dir": str(output_dir),
            "model_name": args.model_name,
            "requested_device": args.device,
            "clinical_warning": CLINICAL_WARNING,
            "source_inspection": None,
            "coordinate_probe": None,
            "tiatoolbox_merge_probe": None,
            "direct_vs_stitched_consistency": None,
            "offset_search": None,
            "artifacts": {},
        }

        row: dict[str, str] | None = None
        if args.run_coordinate_probe or args.run_secondary_consistency_check or args.run_offset_search:
            _patch_path, row = selected_patch_path(selection_dir, args.patch_index)

        if args.run_source_inspection:
            source_report = inspect_tiatoolbox_output_placement(model_name=args.model_name)
            report["source_inspection"] = source_report
            source_path = write_json(source_report, output_dir / "source_inspection.json")
            report["artifacts"]["source_inspection_json"] = str(source_path)
            print(
                "[OK] Source inspection: "
                f"{source_report.get('source_code_conclusion')} ({source_path})"
            )

        if args.run_coordinate_probe:
            coords = selected_patch_coordinates(row or {})
            coordinate_report = run_coordinate_probe(
                x_level0=coords.get("x_level0"),
                y_level0=coords.get("y_level0"),
                patch_size=coords.get("patch_size") or 1024,
            )
            report["coordinate_probe"] = coordinate_report
            coordinate_path = write_json(coordinate_report, output_dir / "coordinate_probe.json")
            report["artifacts"]["coordinate_probe_json"] = str(coordinate_path)
            print(
                "[OK] Coordinate probe: "
                f"{coordinate_report.get('coordinate_probe_conclusion')} ({coordinate_path})"
            )

        if args.run_tiatoolbox_merge_probe:
            merge_report = run_tiatoolbox_merge_probe(output_dir)
            report["tiatoolbox_merge_probe"] = merge_report
            merge_path = write_json(merge_report, output_dir / "tiatoolbox_merge_probe.json")
            report["artifacts"]["tiatoolbox_merge_probe_json"] = str(merge_path)
            print(
                "[OK] TIAToolbox merge probe: "
                f"{merge_report.get('merge_probe_conclusion')} ({merge_path})"
            )

        if args.run_secondary_consistency_check:
            secondary_report = run_secondary_consistency_check(
                selection_dir=selection_dir,
                patch_index=args.patch_index,
                output_dir=output_dir,
                model_name=args.model_name,
                device=args.device,
                root_dir=ROOT_DIR,
            )
            report["direct_vs_stitched_consistency"] = secondary_report
            secondary_path = write_json(
                secondary_report,
                output_dir / "direct_vs_stitched_consistency.json",
            )
            report["artifacts"]["direct_vs_stitched_consistency_json"] = str(secondary_path)
            print(
                "[OK] Secondary direct-vs-stitched check: "
                f"{secondary_report.get('status')} ({secondary_path})"
            )

        if args.run_offset_search:
            secondary_report = report.get("direct_vs_stitched_consistency")
            direct_path = None
            if isinstance(secondary_report, dict):
                direct_path_text = secondary_report.get("direct_prediction_labels_raw_npy")
                if direct_path_text:
                    direct_path = Path(str(direct_path_text))
            if direct_path is None:
                candidate_direct = (
                    output_dir
                    / "secondary_consistency"
                    / "direct_patch_inference"
                    / "prediction_labels_raw.npy"
                )
                direct_path = candidate_direct if candidate_direct.exists() else None

            stitched_path = output_dir.parent / "alignment_probe" / "stitched_prediction_1024.npy"
            if direct_path is None or not direct_path.exists() or not stitched_path.exists():
                offset_report = {
                    "status": "skipped",
                    "test_type": "offset_search",
                    "reason": (
                        "Offset search requires direct prediction labels and an existing "
                        "context-stitch alignment prediction."
                    ),
                    "direct_prediction_labels_raw_npy": str(direct_path) if direct_path else None,
                    "stitched_prediction_npy": str(stitched_path),
                    "clinical_warning": CLINICAL_WARNING,
                }
                write_json(offset_report, output_dir / "offset_search_summary.json")
            else:
                offset_report = run_offset_search(
                    direct_prediction_path=direct_path,
                    stitched_prediction_path=stitched_path,
                    output_dir=output_dir,
                    step=args.offset_search_step,
                )
            report["offset_search"] = offset_report
            report["artifacts"]["offset_search_summary_json"] = str(
                output_dir / "offset_search_summary.json"
            )
            if (output_dir / "offset_search.csv").exists():
                report["artifacts"]["offset_search_csv"] = str(output_dir / "offset_search.csv")
            if (output_dir / "offset_agreement_heatmap.png").exists():
                report["artifacts"]["offset_agreement_heatmap_png"] = str(
                    output_dir / "offset_agreement_heatmap.png"
                )
            print(f"[OK] Offset search: {offset_report.get('status')}")

        overall = determine_overall_conclusion(
            report.get("source_inspection"),
            report.get("coordinate_probe"),
            report.get("tiatoolbox_merge_probe"),
        )
        report["overall_conclusion"] = overall
        report["conclusion_scope"] = (
            "Primary conclusion is based on source/config inspection and coordinate probes. "
            "Secondary direct-vs-stitched checks do not establish clinical validity."
        )

        report_json_path = write_json(
            report,
            output_dir / "tiatoolbox_output_placement_report.json",
        )
        report_md_path = write_output_placement_report_md(
            report,
            output_dir / "tiatoolbox_output_placement_report.md",
        )
        report["artifacts"]["tiatoolbox_output_placement_report_json"] = str(report_json_path)
        report["artifacts"]["tiatoolbox_output_placement_report_md"] = str(report_md_path)
        # Re-write after artifact paths are complete.
        write_json(report, report_json_path)

        summary = {
            "status": "completed",
            "overall_conclusion": overall,
            "source_code_conclusion": (
                report.get("source_inspection") or {}
            ).get("source_code_conclusion"),
            "coordinate_probe_conclusion": (
                report.get("coordinate_probe") or {}
            ).get("coordinate_probe_conclusion"),
            "merge_probe_conclusion": (
                report.get("tiatoolbox_merge_probe") or {}
            ).get("merge_probe_conclusion"),
            "secondary_check_status": (
                report.get("direct_vs_stitched_consistency") or {}
            ).get("status"),
            "offset_search_status": (report.get("offset_search") or {}).get("status"),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
            "clinical_warning": CLINICAL_WARNING,
        }
        summary_path = write_json(summary, output_dir / "output_placement_summary.json")
        print(f"[OK] Overall conclusion: {overall}")
        print(f"[OK] Report JSON: {report_json_path}")
        print(f"[OK] Report MD: {report_md_path}")
        print(f"[OK] Summary: {summary_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = _write_failure(output_dir, str(exc))
        print(f"[FAIL] {exc}")
        print(f"[FAIL] Failure summary: {failure_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
