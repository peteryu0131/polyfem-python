"""Compare a generated example backend run against its source JSON run.

This is a manual diagnostic tool for generated example backend coverage. It
does two comparisons:

1. generated example output vs. source JSON output
2. generated example output vs. the source JSON ``tests`` block

By default only the first comparison is required to pass. Use
``--require-tests-match`` after the PolyFEM backend and polyfem-data tests are
known to be synchronized.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE = (
    ROOT
    / "examples"
    / "classic_example"
    / "2D"
    / "contact_2d_golf_ball_deformable_wall_generated_api.py"
)
DEFAULT_SOURCE_JSON = (
    ROOT
    / "polyfem-data"
    / "contact"
    / "examples"
    / "2D"
    / "golf-ball-doformable-wall.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "generated-example-backend-check"


@dataclass(frozen=True)
class MetricDiff:
    key: str
    left: float
    right: float
    diff: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.diff <= self.tolerance


def _error_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in payload.items()
        if key.startswith("err_") and isinstance(value, (int, float))
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def load_output_metrics(output_dir: Path) -> dict[str, float]:
    sim_json = output_dir / "sim.json"
    if not sim_json.exists():
        raise FileNotFoundError(f"backend output is missing {sim_json}")
    return _error_metrics(_load_json(sim_json))


def load_test_metrics(source_json: Path) -> tuple[dict[str, float], float]:
    payload = _load_json(source_json)
    tests = payload.get("tests")
    if not isinstance(tests, dict):
        return {}, 1e-8
    tolerance = float(tests.get("margin", 1e-8))
    return _error_metrics(tests), tolerance


def compare_metric_maps(
    *,
    left: dict[str, float],
    right: dict[str, float],
    tolerance: float,
) -> list[MetricDiff]:
    keys = sorted(set(left) | set(right))
    rows: list[MetricDiff] = []
    for key in keys:
        if key not in left or key not in right:
            rows.append(MetricDiff(key, left.get(key, float("nan")), right.get(key, float("nan")), float("inf"), tolerance))
            continue
        diff = abs(left[key] - right[key])
        rows.append(MetricDiff(key, left[key], right[key], diff, tolerance))
    return rows


def _print_rows(title: str, rows: Sequence[MetricDiff]) -> bool:
    print(title)
    all_passed = True
    for row in rows:
        status = "PASS" if row.passed else "FAIL"
        all_passed = all_passed and row.passed
        print(
            f"  {status} {row.key}: "
            f"left={row.left:.17g} right={row.right:.17g} "
            f"diff={row.diff:.3g} tolerance={row.tolerance:.3g}"
        )
    return all_passed


def _import_example(example_path: Path):
    if str(example_path.parent) not in sys.path:
        sys.path.insert(0, str(example_path.parent))
    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_generated_example(example_path: Path, output_root: Path) -> Path:
    from polyfempy.runtime import solve

    module = _import_example(example_path)
    config_for_workspace = getattr(module, "config_for_workspace", None)
    if not callable(config_for_workspace):
        raise RuntimeError(
            f"{example_path} must expose config_for_workspace(workspace)"
        )

    workspace = output_root / "generated" / example_path.stem
    workspace.mkdir(parents=True, exist_ok=True)
    cfg = config_for_workspace(workspace)
    solve(cfg=cfg)
    return workspace


def run_source_json(
    source_json: Path,
    output_root: Path,
    *,
    log_level: int,
    max_threads: int,
) -> Path:
    workspace = output_root / "source-json" / source_json.stem
    workspace.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "polyfempy",
        "solve",
        str(source_json),
        "--output-dir",
        str(workspace),
        "--log-level",
        str(log_level),
        "--max-threads",
        str(max_threads),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "source JSON backend run failed with exit code "
            f"{result.returncode}: {' '.join(command)}"
        )
    return workspace


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a generated example and its source JSON through the backend, "
            "then compare their err_* metrics."
        )
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=DEFAULT_EXAMPLE,
        help="generated example Python file to run",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=DEFAULT_SOURCE_JSON,
        help="polyfem-data source JSON mirrored by the generated example",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory for generated and source JSON backend outputs",
    )
    parser.add_argument(
        "--generated-source-tolerance",
        type=float,
        default=1e-8,
        help="required tolerance for generated output vs source JSON output",
    )
    parser.add_argument(
        "--require-tests-match",
        action="store_true",
        help=(
            "also fail if generated output does not match the source JSON "
            "tests block"
        ),
    )
    parser.add_argument("--log-level", type=int, default=2)
    parser.add_argument("--max-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    example_path = args.example.resolve()
    source_json = args.source_json.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"generated example: {example_path}")
    print(f"source JSON:       {source_json}")
    print(f"output root:       {output_root}")

    generated_output = run_generated_example(example_path, output_root)
    source_output = run_source_json(
        source_json,
        output_root,
        log_level=args.log_level,
        max_threads=args.max_threads,
    )

    generated_metrics = load_output_metrics(generated_output)
    source_metrics = load_output_metrics(source_output)
    test_metrics, test_tolerance = load_test_metrics(source_json)

    print(f"generated output:  {generated_output}")
    print(f"source output:     {source_output}")

    generated_source_rows = compare_metric_maps(
        left=generated_metrics,
        right=source_metrics,
        tolerance=args.generated_source_tolerance,
    )
    generated_matches_source = _print_rows(
        "generated output vs source JSON output:",
        generated_source_rows,
    )

    generated_matches_tests = True
    if test_metrics:
        tests_rows = compare_metric_maps(
            left=generated_metrics,
            right=test_metrics,
            tolerance=test_tolerance,
        )
        generated_matches_tests = _print_rows(
            "generated output vs source JSON tests:",
            tests_rows,
        )
    else:
        print("source JSON has no tests block; skipping tests comparison")

    if not generated_matches_source:
        return 1
    if args.require_tests_match and not generated_matches_tests:
        return 1
    if not generated_matches_tests:
        print(
            "WARNING: generated output matches the source JSON run, but does "
            "not match the source JSON tests block."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
