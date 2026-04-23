from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

from .report import summarize_history_bundle
from .solve import solve


def make_timestamped_workspace(base_dir: Path | str, tag: str) -> Path:
    """Create ``<base_dir>/<tag>_<unix_ts>`` and return the resolved path."""
    workspace_root = Path(base_dir).resolve()
    workspace = workspace_root / f"{tag}_{int(time.time())}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace.resolve()


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _format_table(columns: list[str], rows: list[dict[str, Any]]) -> list[str]:
    rendered = [{col: _format_scalar(row.get(col, "")) for col in columns} for row in rows]
    widths = {
        col: max(len(col), *(len(row[col]) for row in rendered)) if rendered else len(col)
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    separator = "  ".join("-" * widths[col] for col in columns)
    lines = [header, separator]
    for row in rendered:
        lines.append("  ".join(row[col].ljust(widths[col]) for col in columns))
    return lines


def _write_csv(path: Path, *, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def _bundle_body_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for body_id in sorted(bundle.get("body_legend", {})):
        row = dict(bundle["body_legend"][body_id])
        row["body_id"] = body_id
        rows.append(row)
    return rows


def format_history_summary(bundle: dict[str, Any]) -> str:
    if not bundle.get("available", False):
        return (
            "# PolyFEM History Summary\n"
            "history_available: false\n"
            "note: result.history is empty\n"
        )

    lines = [
        "# PolyFEM History Summary",
        f"history_source: {bundle.get('history_source', 'unknown')}",
        (
            "fields: "
            f"u={bundle['fields']['u']} "
            f"von_mises={bundle['fields']['von_mises']} "
            f"stress={bundle['fields']['stress']}"
        ),
    ]

    body_rows = _bundle_body_rows(bundle)
    if body_rows:
        lines.extend(["", "[bodies]"])
        lines.extend(
            _format_table(
                ["body_id", "geometry_index", "volume_selection", "mesh_stem"],
                body_rows,
            )
        )

    per_body_rows = list(bundle.get("steps_by_body", []))
    if per_body_rows:
        lines.extend(["", "[steps_by_body]"])
        lines.extend(
            _format_table(
                ["step", "time", "body_id", "mesh_stem", "n_points", "u_max", "vm_mean", "vm_max", "vm_p95", "stress_abs_max"],
                per_body_rows,
            )
        )
        lines.append("")
        lines.append(f"history_steps_by_body.csv rows: {len(per_body_rows)}")

    return "\n".join(lines) + "\n"


def write_history_artifacts(*, result, workspace: Path, cfg) -> dict[str, Path]:
    bundle = summarize_history_bundle(result, cfg=cfg)

    summary_txt_path = (workspace / "history_summary.txt").resolve()
    summary_txt_path.write_text(format_history_summary(bundle), encoding="utf-8")

    bundle_json_path = (workspace / "history_bundle.json").resolve()
    bundle_json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    paths = {
        "summary_txt": summary_txt_path,
        "bundle_json": bundle_json_path,
    }

    if bundle.get("available", False):
        body_rows = _bundle_body_rows(bundle)
        if body_rows:
            paths["body_legend_csv"] = _write_csv(
                workspace / "history_body_legend.csv",
                columns=["body_id", "geometry_index", "volume_selection", "mesh_stem"],
                rows=body_rows,
            )
        if bundle.get("steps_by_body"):
            paths["steps_by_body_csv"] = _write_csv(
                workspace / "history_steps_by_body.csv",
                columns=[
                    "step",
                    "time",
                    "body_id",
                    "geometry_index",
                    "volume_selection",
                    "mesh_stem",
                    "n_points",
                    "u_max",
                    "vm_mean",
                    "vm_max",
                    "vm_p95",
                    "stress_abs_max",
                ],
                rows=list(bundle["steps_by_body"]),
            )

    return paths


def report_history_bundle(*, result, workspace: Path, cfg, elapsed: Optional[float] = None) -> Path:
    bundle_path = result.write_history_bundle_txt(workspace / "history_bundle.txt", cfg=cfg)
    artifact_paths = write_history_artifacts(result=result, workspace=workspace, cfg=cfg)
    summary_text = artifact_paths["summary_txt"].read_text(encoding="utf-8")

    if elapsed is not None:
        print(f"solve() took {elapsed:.2f}s\n")
    print(summary_text, end="")
    print(f"\nhistory bundle: {bundle_path}")
    print(f"history summary: {artifact_paths['summary_txt']}")
    print(f"history json: {artifact_paths['bundle_json']}")
    if "steps_by_body_csv" in artifact_paths:
        print(f"history steps by body csv: {artifact_paths['steps_by_body_csv']}")
    if "body_legend_csv" in artifact_paths:
        print(f"history body legend csv: {artifact_paths['body_legend_csv']}")
    print(f"\nworkspace: {workspace}")
    return bundle_path


def emit_history_bundle(*, result, workspace: Path, cfg, elapsed: float) -> Path:
    return report_history_bundle(result=result, workspace=workspace, cfg=cfg, elapsed=elapsed)


def solve_and_report(*, cfg, workspace: Path):
    t0 = time.perf_counter()
    result = solve(cfg=cfg)
    elapsed = time.perf_counter() - t0
    report_history_bundle(result=result, workspace=workspace, cfg=cfg, elapsed=elapsed)
    return result
