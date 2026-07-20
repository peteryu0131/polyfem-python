from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "generate_polyfem_api.py"
GENERATOR_SCRIPT = ROOT / "python-from-jse" / "tools" / "generate_with_overrides.py"
GENERATED_DIR = ROOT / "polyfempy" / "generated"


def import_workflow(module_name: str = "generate_polyfem_api_for_test"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneratePolyfemApiWorkflowTests(unittest.TestCase):
    def test_default_workflow_runs_generator_from_repo_root(self):
        workflow = import_workflow()

        with mock.patch.object(workflow.subprocess, "run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()):
                result = workflow.main([])

        self.assertEqual(0, result)
        self.assertEqual(
            [
                [
                    sys.executable,
                    str(GENERATOR_SCRIPT),
                    "--schema-file",
                    str(ROOT / "python-from-jse" / "json-specs" / "input-spec.json"),
                    "--output-file",
                    str(GENERATED_DIR / "generated_class.py"),
                    "--api-output-file",
                    str(GENERATED_DIR / "generated_api.py"),
                    "--manifest-dir",
                    str(GENERATED_DIR),
                    "--relationships",
                    str(ROOT / "generator-config" / "id_relationships.json"),
                    "--api-aliases",
                    str(ROOT / "generator-config" / "api_aliases.json"),
                    "--model-entry",
                    "polyfem.model",
                ],
            ],
            [call.args[0] for call in run_mock.call_args_list],
        )
        self.assertEqual(
            [ROOT],
            [call.kwargs["cwd"] for call in run_mock.call_args_list],
        )

    def test_check_workflow_runs_backend_free_checks_after_generation(self):
        workflow = import_workflow()

        with mock.patch.object(workflow.subprocess, "run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()):
                result = workflow.main(["--check"])

        self.assertEqual(0, result)
        self.assertEqual(
            [
                [
                    sys.executable,
                    str(GENERATOR_SCRIPT),
                    "--schema-file",
                    str(ROOT / "python-from-jse" / "json-specs" / "input-spec.json"),
                    "--output-file",
                    str(GENERATED_DIR / "generated_class.py"),
                    "--api-output-file",
                    str(GENERATED_DIR / "generated_api.py"),
                    "--manifest-dir",
                    str(GENERATED_DIR),
                    "--relationships",
                    str(ROOT / "generator-config" / "id_relationships.json"),
                    "--api-aliases",
                    str(ROOT / "generator-config" / "api_aliases.json"),
                    "--model-entry",
                    "polyfem.model",
                ],
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_examples_public_surface.py",
                    "tests/test_generated_api_example.py",
                ],
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                ],
            ],
            [call.args[0] for call in run_mock.call_args_list],
        )
        self.assertEqual(
            [ROOT, ROOT, ROOT / "python-from-jse"],
            [call.kwargs["cwd"] for call in run_mock.call_args_list],
        )

    def test_workflow_reports_missing_required_directory(self):
        workflow = import_workflow()

        def fake_exists(path: Path) -> bool:
            return path != ROOT / "python-from-jse"

        with (
            mock.patch.object(workflow.Path, "exists", fake_exists),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = workflow.main([])

        self.assertEqual(1, result)
        self.assertIn("Missing required path", stderr.getvalue())
        self.assertIn("python-from-jse", stderr.getvalue())
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
