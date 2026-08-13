"""Run generated contact examples listed by PolyFEM contact test files.

This is a thin batch wrapper around ``tools/check_generated_example_backend.py``.
It keeps the single-case checker as the source of truth and only adds:

- reading ``polyfem/tests/contact_2d.txt`` and ``contact_3d.txt``
- mapping each source JSON to its generated example
- saving per-case logs plus machine-readable and text summaries
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
CONTACT_LISTS = [
    ROOT / "polyfem" / "tests" / "contact_2d.txt",
    ROOT / "polyfem" / "tests" / "contact_3d.txt",
]
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "generated-contact-backend-check"

TARGET_NAME_OVERRIDES = {
    "2D/golf-ball-doformable-wall.json": (
        "contact_2d_golf_ball_deformable_wall_generated_api.py"
    ),
    "3D/friction/high-school-physics-slopetest-mu=0.50.json": (
        "contact_3d_friction_high_school_slopetest_generated_api.py"
    ),
}


@dataclass(frozen=True)
class ContactCase:
    list_file: str
    line_number: int
    source_rel: str
    source_json: str
    generated_example: str


@dataclass(frozen=True)
class CaseResult:
    source_rel: str
    generated_example: str
    status: str
    returncode: int
    log_file: str


def slugify(value: str) -> str:
    import re

    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", value)).strip("_").lower()


def generated_example_for_source(source_rel: str) -> Path:
    if not source_rel.startswith("contact/examples/"):
        raise ValueError(f"unsupported source path: {source_rel}")

    contact_relative = Path(source_rel.removeprefix("contact/examples/"))
    dim = contact_relative.parts[0]
    source_key = contact_relative.as_posix()
    override = TARGET_NAME_OVERRIDES.get(source_key)
    if override is not None:
        return ROOT / "examples" / "classic_example" / dim / override

    relative_without_suffix = contact_relative.relative_to(dim).with_suffix("")
    name = f"contact_{dim.lower()}_{slugify('_'.join(relative_without_suffix.parts))}_generated_api.py"
    return ROOT / "examples" / "classic_example" / dim / name


def iter_active_contact_cases(list_paths: Sequence[Path]) -> list[ContactCase]:
    cases: list[ContactCase] = []
    for list_path in list_paths:
        for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
            source_rel = line.strip()
            if not source_rel or source_rel.startswith("#"):
                continue
            if not source_rel.startswith("contact/examples/"):
                continue

            source_json = ROOT / "polyfem-data" / source_rel
            generated_example = generated_example_for_source(source_rel)
            cases.append(
                ContactCase(
                    list_file=str(list_path.relative_to(ROOT)),
                    line_number=line_number,
                    source_rel=source_rel,
                    source_json=str(source_json),
                    generated_example=str(generated_example),
                )
            )
    return cases


def safe_case_name(source_rel: str) -> str:
    return slugify(source_rel.removeprefix("contact/examples/").removesuffix(".json"))


def run_case(
    case: ContactCase,
    output_root: Path,
    *,
    generated_source_tolerance: float,
    require_tests_match: bool,
    keep_visual_output: bool,
    log_level: int,
    max_threads: int,
) -> CaseResult:
    case_name = safe_case_name(case.source_rel)
    case_root = output_root / "runs" / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    log_file = output_root / "logs" / f"{case_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(ROOT / "tools" / "check_generated_example_backend.py"),
        "--example",
        case.generated_example,
        "--source-json",
        case.source_json,
        "--output-root",
        str(case_root),
        "--generated-source-tolerance",
        str(generated_source_tolerance),
        "--log-level",
        str(log_level),
        "--max-threads",
        str(max_threads),
    ]
    if require_tests_match:
        command.append("--require-tests-match")
    if keep_visual_output:
        command.append("--keep-visual-output")

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log_file.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                "STDOUT:",
                result.stdout,
                "",
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )

    return CaseResult(
        source_rel=case.source_rel,
        generated_example=str(Path(case.generated_example).relative_to(ROOT)),
        status="PASS" if result.returncode == 0 else "FAIL",
        returncode=result.returncode,
        log_file=str(log_file.relative_to(ROOT)),
    )


def write_summaries(
    output_root: Path,
    cases: Sequence[ContactCase],
    results: Sequence[CaseResult],
) -> None:
    passed = sum(1 for result in results if result.status == "PASS")
    failed = len(results) - passed
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "cases": [asdict(result) for result in results],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Generated contact backend check summary",
        f"Total:  {len(results)}",
        f"PASS:   {passed}",
        f"FAIL:   {failed}",
        "",
        "| Status | Source JSON | Generated example | Log |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.status} | `{result.source_rel}` | "
            f"`{result.generated_example}` | `{result.log_file}` |"
        )

    if len(cases) != len(results):
        lines.extend(["", f"Stopped early after {len(results)} of {len(cases)} cases."])

    (output_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generated backend checks for active PolyFEM contact tests."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--match", default=None, help="only run cases containing this text")
    parser.add_argument("--generated-source-tolerance", type=float, default=1e-5)
    parser.add_argument("--require-tests-match", action="store_true")
    parser.add_argument("--keep-visual-output", action="store_true")
    parser.add_argument("--log-level", type=int, default=2)
    parser.add_argument("--max-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases = iter_active_contact_cases(CONTACT_LISTS)
    if args.match:
        cases = [case for case in cases if args.match in case.source_rel]
    if args.limit is not None:
        cases = cases[: args.limit]

    results: list[CaseResult] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.source_rel}")
        result = run_case(
            case,
            output_root,
            generated_source_tolerance=args.generated_source_tolerance,
            require_tests_match=args.require_tests_match,
            keep_visual_output=args.keep_visual_output,
            log_level=args.log_level,
            max_threads=args.max_threads,
        )
        results.append(result)
        print(f"  {result.status} log={result.log_file}")
        write_summaries(output_root, cases, results)

    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
