"""Generate PolyFEM's packaged Python API from the repo root."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "python-from-jse"
GENERATOR_CONFIG_DIR = REPO_ROOT / "generator-config"
POLYFEM_SCHEMA_FILE = GENERATOR_ROOT / "json-specs" / "input-spec.json"
POLYFEMPY_PACKAGE_DIR = REPO_ROOT / "polyfempy"
GENERATED_DIR = POLYFEMPY_PACKAGE_DIR / "generated_api"


def workflow_steps(run_checks: bool) -> list[tuple[list[str], Path]]:
    steps = [
        (
            [
                sys.executable,
                str(GENERATOR_ROOT / "tools" / "generate_with_overrides.py"),
                "--schema-file",
                str(POLYFEM_SCHEMA_FILE),
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
            ],
            REPO_ROOT,
        ),
    ]
    if run_checks:
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
            ),
        ])
    return steps


def missing_required_paths() -> list[Path]:
    required = [
        GENERATOR_ROOT,
        GENERATOR_ROOT / "tools" / "generate_with_overrides.py",
        POLYFEM_SCHEMA_FILE,
        GENERATOR_CONFIG_DIR,
        GENERATOR_CONFIG_DIR / "api_aliases.json",
        GENERATOR_CONFIG_DIR / "id_relationships.json",
        POLYFEMPY_PACKAGE_DIR,
    ]
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
    args = parser.parse_args(argv)

    missing = missing_required_paths()
    if missing:
        for path in missing:
            print(f"Missing required path: {path}", file=sys.stderr)
        print(
            "If this is a fresh checkout, make sure python-from-jse and "
            "generator-config are present.",
            file=sys.stderr,
        )
        return 1

    for command, cwd in workflow_steps(args.check):
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=cwd)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
