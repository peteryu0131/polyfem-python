"""Generate PolyFEM's packaged Python API from the repo root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "python-from-jse"
GENERATOR_CONFIG_DIR = REPO_ROOT / "generator-config"
DEFAULT_POLYFEM_SOURCE_DIR = REPO_ROOT / "external" / "polyfem"
POLYFEMPY_PACKAGE_DIR = REPO_ROOT / "polyfempy"
GENERATED_DIR = POLYFEMPY_PACKAGE_DIR / "generated_api"
LINKED_SOLVER_SPEC_FILES = (
    "linear-solver-spec.json",
    "nonlinear-solver-spec.json",
)


def polyfem_schema_file(polyfem_source_dir: Path) -> Path:
    return polyfem_source_dir / "json-specs" / "input-spec.json"


def workflow_steps(
    run_checks: bool,
    *,
    schema_file: Path,
    include_spec_dirs: list[Path] | None = None,
) -> list[tuple[list[str], Path, dict[str, str] | None]]:
    include_spec_dirs = include_spec_dirs or []
    generator_command = [
        sys.executable,
        str(GENERATOR_ROOT / "tools" / "generate_with_overrides.py"),
        "--schema-file",
        str(schema_file),
    ]
    for include_spec_dir in include_spec_dirs:
        generator_command.extend(["--include-spec-dir", str(include_spec_dir)])
    generator_command.extend([
        "--output-file",
        str(GENERATED_DIR / "generated_class.py"),
        "--api-output-file",
        str(GENERATED_DIR / "generated_api.py"),
        "--manifest-dir",
        str(GENERATED_DIR),
        "--relationships",
        str(GENERATOR_CONFIG_DIR / "id_relationships.json"),
        "--api-aliases",
        str(GENERATOR_CONFIG_DIR / "api_aliases.json"),
        "--model-entry",
        "polyfem.model",
    ])
    steps = [
        (
            generator_command,
            REPO_ROOT,
            None,
        ),
    ]
    if run_checks:
        generator_test_env = {
            "POLYFEM_SPEC_DIR": str(schema_file.parent),
            "POLYFEM_INCLUDE_SPEC_DIRS": os.pathsep.join(
                str(path) for path in include_spec_dirs
            ),
        }
        steps.extend([
            (
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_examples_public_surface.py",
                    "tests/test_generated_api_example.py",
                ],
                REPO_ROOT,
                None,
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                ],
                GENERATOR_ROOT,
                generator_test_env,
            ),
        ])
    return steps


def missing_required_paths(schema_file: Path, include_spec_dirs: list[Path]) -> list[Path]:
    required = [
        GENERATOR_ROOT,
        GENERATOR_ROOT / "tools" / "generate_with_overrides.py",
        schema_file,
        GENERATOR_CONFIG_DIR,
        GENERATOR_CONFIG_DIR / "api_aliases.json",
        GENERATOR_CONFIG_DIR / "id_relationships.json",
        POLYFEMPY_PACKAGE_DIR,
    ]
    for include_spec_dir in include_spec_dirs:
        required.extend([
            include_spec_dir,
        ])
    search_dirs = [schema_file.parent, *include_spec_dirs]
    for spec_file in LINKED_SOLVER_SPEC_FILES:
        if not any((search_dir / spec_file).exists() for search_dir in search_dirs):
            required.append(schema_file.parent / spec_file)
    return [path for path in required if not path.exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate polyfempy/generated_api from python-from-jse and "
            "generator-config."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run backend-free generated API and example parity checks.",
    )
    parser.add_argument(
        "--polyfem-source-dir",
        default=DEFAULT_POLYFEM_SOURCE_DIR,
        type=Path,
        help=(
            "Path to a PolyFEM source checkout. Defaults to the "
            "external/polyfem submodule."
        ),
    )
    parser.add_argument(
        "--include-spec-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional directory to search for included spec files. "
            "Can be passed more than once."
        ),
    )
    args = parser.parse_args(argv)

    schema_file = polyfem_schema_file(args.polyfem_source_dir)
    include_spec_dirs = args.include_spec_dir
    missing = missing_required_paths(schema_file, include_spec_dirs)
    if missing:
        for path in missing:
            print(f"Missing required path: {path}", file=sys.stderr)
        print(
            "If this is a fresh checkout, make sure python-from-jse and "
            "generator-config are present, then run "
            "`git submodule update --init --recursive` for external/polyfem. "
            "If the missing file is a linked solver spec, pass "
            "`--include-spec-dir <dir-containing-linked-specs>`.",
            file=sys.stderr,
        )
        return 1

    for command, cwd, extra_env in workflow_steps(
        args.check,
        schema_file=schema_file,
        include_spec_dirs=include_spec_dirs,
    ):
        print("+ " + " ".join(command), flush=True)
        run_kwargs = {"cwd": cwd}
        if extra_env is not None:
            env = os.environ.copy()
            env.update(extra_env)
            run_kwargs["env"] = env
        result = subprocess.run(command, **run_kwargs)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
