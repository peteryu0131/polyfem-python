from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Literal, Optional

from .config import Output
from .report import summarize_history_bundle
from .solve import solve


LogLevelName = Literal["trace", "debug", "info", "warn", "warning", "error", "critical", "off"]


def make_timestamped_workspace(base_dir: Path | str, tag: str) -> Path:
    """Create ``<base_dir>/<tag>_<unix_ts>`` and return the resolved path."""
    workspace_root = Path(base_dir).resolve()
    workspace = workspace_root / f"{tag}_{int(time.time())}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace.resolve()


def _ensure_output(cfg: Any) -> Output:
    output = getattr(cfg, "output", None)
    if output is None:
        output = Output()
        cfg.output = output
        return output
    if isinstance(output, dict):
        output = Output.from_dict(output)
        cfg.output = output
        return output
    return output


def terminal_log(
    cfg: Any,
    *,
    level: int | LogLevelName = "debug",
    file_level: int | LogLevelName = "debug",
    path: str = "polyfem.log",
    print_terminal: bool = True,
    quiet: Optional[bool] = None,
) -> Any:
    """Configure terminal + file logging on ``cfg.output`` in one call.

    Typical use:
        ``terminal_log(cfg)``

    This means:
    - terminal log level = ``"debug"``
    - file log level = ``"debug"``
    - log file path = ``"polyfem.log"``
    - terminal output is printed

    Args:
        cfg:
            The simulation config returned by ``build_config(...)``.
        level:
            Terminal / console log level. Can be an integer level or one of:
            ``"trace"``, ``"debug"``, ``"info"``, ``"warn"``,
            ``"warning"``, ``"error"``, ``"critical"``, ``"off"``.
            Default: ``"debug"``.
        file_level:
            Log level for the log file. Same accepted values as ``level``.
            Default: ``"debug"``.
        path:
            Output log filename or path. Default: ``"polyfem.log"``.
        print_terminal:
            If ``True``, print PolyFEM logs to the terminal. If ``False``,
            keep logs in the file but silence/minimize terminal output.
            Default: ``True``.
        quiet:
            Backward-compatible alias for ``not print_terminal``.

    Returns:
        The same ``cfg`` object, so calls can stay lightweight and chainable.
    """
    if quiet is None:
        quiet = not bool(print_terminal)
    elif bool(quiet) == bool(print_terminal):
        raise ValueError(
            "terminal_log got conflicting print_terminal and quiet values: "
            f"print_terminal={print_terminal!r}, quiet={quiet!r}"
        )

    output = _ensure_output(cfg)
    output.set_log(
        path=path,
        level=level,
        file_level=file_level,
        quiet=quiet,
    )
    return cfg


def result_output(
    cfg: Any,
    *,
    directory: str = ".",
    json_name: str = "impact_stats.json",
    pvd_name: str = "impact.pvd",
    save_vtu: bool = True,
    save_time_sequence: bool = True,
    timestep_prefix: str = "impact_step_",
    vismesh_rel_area: float | None = 10_000_000,
    surface: bool = False,
    wireframe: bool = False,
    points: bool = False,
    material: bool = True,
    body_ids: bool = True,
    velocity: bool = True,
    acceleration: bool | None = None,
    scalar_values: bool = True,
    tensor_values: bool = True,
) -> Any:
    """Configure the common JSON + ParaView + history outputs on ``cfg.output``.

    Typical use:
        ``result_output(cfg)``

    This means:
    - write outputs into the workspace directory (``directory="."``)
    - write JSON stats to ``impact_stats.json``
    - write ParaView sequence root file ``impact.pvd``
    - export step VTU files
    - save the time sequence
    - use ``impact_step_`` as the step filename prefix
    - enable common ParaView fields such as ``material``, ``body_ids``,
      ``velocity``, ``scalar_values`` and ``tensor_values``

    Args:
        cfg:
            The simulation config returned by ``build_config(...)``.
        directory:
            Output directory, relative to the run workspace unless absolute.
            Default: ``"."``.
        json_name:
            JSON output filename. Default: ``"impact_stats.json"``.
        pvd_name:
            ParaView ``.pvd`` filename. Default: ``"impact.pvd"``.
        save_vtu:
            Whether to export step ``.vtu`` files. Default: ``True``.
        save_time_sequence:
            Whether to save a time sequence for transient runs. Default:
            ``True``.
        timestep_prefix:
            Prefix for step outputs such as ``impact_step_0.vtu``.
            Default: ``"impact_step_"``.
        vismesh_rel_area:
            ParaView visualization-mesh relative area control. Default:
            ``10_000_000``.
        surface:
            Enable surface output in ParaView. Default: ``False``.
        wireframe:
            Enable wireframe output in ParaView. Default: ``False``.
        points:
            Enable point-cloud output in ParaView. Default: ``False``.
        material:
            Export material ids in ParaView fields. Default: ``True``.
        body_ids:
            Export body ids in ParaView fields. Default: ``True``.
        velocity:
            Export velocity field in ParaView. Default: ``True``.
        acceleration:
            Export acceleration field in ParaView. ``None`` means “leave this
            toggle untouched unless explicitly requested”. Default: ``None``.
        scalar_values:
            Export scalar-valued fields. Default: ``True``.
        tensor_values:
            Export tensor-valued fields. Default: ``True``.

    Returns:
        The same ``cfg`` object, so calls can stay lightweight and chainable.
    """
    output = _ensure_output(cfg)
    output.directory = directory
    output.json = json_name
    output.configure_vtu_export(save_vtu)
    output.set_paraview_sequence(
        file_name=pvd_name,
        surface=surface,
        wireframe=wireframe,
        points=points,
        vismesh_rel_area=vismesh_rel_area,
    )
    output.set_history_sequence(
        timestep_prefix=timestep_prefix,
        save_time_sequence=save_time_sequence,
    )
    output.enable_paraview_fields(
        material=material,
        body_ids=body_ids,
        velocity=velocity,
        acceleration=acceleration,
        scalar_values=scalar_values,
        tensor_values=tensor_values,
    )
    return cfg


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
        lines.append(f"steps_by_body rows: {len(per_body_rows)}")

    return "\n".join(lines) + "\n"


def write_history_artifacts(
    *,
    result,
    workspace: Path,
    cfg,
    include_summary_txt: bool = True,
    include_bundle_json: bool = True,
    include_steps_by_body_csv: bool = True,
    include_body_legend_csv: bool = False,
) -> dict[str, Path]:
    bundle = summarize_history_bundle(result, cfg=cfg)

    paths: dict[str, Path] = {}

    if include_summary_txt:
        summary_txt_path = (workspace / "history_summary.txt").resolve()
        summary_txt_path.write_text(format_history_summary(bundle), encoding="utf-8")
        paths["summary_txt"] = summary_txt_path

    if include_bundle_json:
        bundle_json_path = (workspace / "history_bundle.json").resolve()
        bundle_json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
        paths["bundle_json"] = bundle_json_path

    if bundle.get("available", False):
        body_rows = _bundle_body_rows(bundle)
        if body_rows and include_body_legend_csv:
            paths["body_legend_csv"] = _write_csv(
                workspace / "history_body_legend.csv",
                columns=["body_id", "geometry_index", "volume_selection", "mesh_stem"],
                rows=body_rows,
            )
        if bundle.get("steps_by_body") and include_steps_by_body_csv:
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


def report_history_bundle(
    *,
    result,
    workspace: Path,
    cfg,
    elapsed: Optional[float] = None,
    started_at: Optional[float] = None,
    include_bundle_txt: bool = False,
    include_summary_txt: bool = True,
    include_bundle_json: bool = True,
    include_steps_by_body_csv: bool = True,
    include_body_legend_csv: bool = False,
    print_terminal: bool = True,
) -> Path:
    """Write compact history/report artifacts and optionally print a summary.

    Default output is intentionally lightweight:
    - ``history_summary.txt`` for humans
    - ``history_bundle.json`` for scripts
    - ``history_steps_by_body.csv`` for tabular analysis

    The older ``history_bundle.txt`` and ``history_body_legend.csv`` remain
    available behind optional flags when needed.
    """
    if elapsed is None and started_at is not None:
        elapsed = time.perf_counter() - started_at

    bundle_path = None
    if include_bundle_txt:
        bundle_path = result.write_history_bundle_txt(workspace / "history_bundle.txt", cfg=cfg)

    artifact_paths = write_history_artifacts(
        result=result,
        workspace=workspace,
        cfg=cfg,
        include_summary_txt=include_summary_txt,
        include_bundle_json=include_bundle_json,
        include_steps_by_body_csv=include_steps_by_body_csv,
        include_body_legend_csv=include_body_legend_csv,
    )
    summary_text = (
        artifact_paths["summary_txt"].read_text(encoding="utf-8")
        if "summary_txt" in artifact_paths
        else format_history_summary(summarize_history_bundle(result, cfg=cfg))
    )

    if print_terminal:
        if elapsed is not None:
            print(f"solve() took {elapsed:.2f}s\n")
        print(summary_text, end="")
        if bundle_path is not None:
            print(f"\nhistory bundle: {bundle_path}")
        if "summary_txt" in artifact_paths:
            print(f"history summary: {artifact_paths['summary_txt']}")
        if "bundle_json" in artifact_paths:
            print(f"history json: {artifact_paths['bundle_json']}")
        if "steps_by_body_csv" in artifact_paths:
            print(f"history steps by body csv: {artifact_paths['steps_by_body_csv']}")
        if "body_legend_csv" in artifact_paths:
            print(f"history body legend csv: {artifact_paths['body_legend_csv']}")
        print(f"\nworkspace: {workspace}")
    if bundle_path is not None:
        return bundle_path
    if "summary_txt" in artifact_paths:
        return artifact_paths["summary_txt"]
    if "bundle_json" in artifact_paths:
        return artifact_paths["bundle_json"]
    return workspace.resolve()


def emit_history_bundle(
    *,
    result,
    workspace: Path,
    cfg,
    elapsed: Optional[float] = None,
    started_at: Optional[float] = None,
    print_terminal: bool = True,
) -> Path:
    return report_history_bundle(
        result=result,
        workspace=workspace,
        cfg=cfg,
        elapsed=elapsed,
        started_at=started_at,
        print_terminal=print_terminal,
    )


def solve_with_timing(*, cfg):
    """Run ``solve(cfg=...)`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = solve(cfg=cfg)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def solve_and_report(*, cfg, workspace: Path):
    """Backward-compatible convenience wrapper around solve + report."""
    result, elapsed = solve_with_timing(cfg=cfg)
    report_history_bundle(result=result, workspace=workspace, cfg=cfg, elapsed=elapsed)
    return result
