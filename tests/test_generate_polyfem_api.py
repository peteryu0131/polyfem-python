from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "generate_polyfem_api.py"
GENERATOR_SCRIPT = ROOT / "python-from-jse" / "tools" / "generate_with_overrides.py"
POLYFEM_SOURCE_DIR = ROOT / "polyfem"
POLYFEM_SCHEMA_FILE = POLYFEM_SOURCE_DIR / "json-specs" / "input-spec.json"
GENERATED_DIR = ROOT / "polyfempy" / "generated_api"


def import_workflow(module_name: str = "generate_polyfem_api_for_test"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneratePolyfemApiWorkflowTests(unittest.TestCase):
    def test_default_workflow_runs_generator_from_repo_root(self):
        workflow = import_workflow()

        with (
            mock.patch.object(workflow, "resolve_include_spec_dirs", return_value=[]),
            mock.patch.object(workflow, "missing_required_paths", return_value=[]),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
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
                    str(POLYFEM_SCHEMA_FILE),
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

    def test_polyfem_source_dir_override_changes_schema_location(self):
        workflow = import_workflow()
        custom_source_dir = ROOT / "custom-polyfem-source"
        custom_schema_file = custom_source_dir / "json-specs" / "input-spec.json"

        with (
            mock.patch.object(workflow, "resolve_include_spec_dirs", return_value=[]),
            mock.patch.object(workflow, "missing_required_paths", return_value=[]),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
            run_mock.return_value = mock.Mock(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()):
                result = workflow.main([
                    "--polyfem-source-dir",
                    str(custom_source_dir),
                ])

        self.assertEqual(0, result)
        self.assertIn(
            str(custom_schema_file),
            run_mock.call_args_list[0].args[0],
        )

    def test_include_spec_dir_is_forwarded_to_generator(self):
        workflow = import_workflow()
        include_dir = ROOT / "custom-linked-specs"

        with (
            mock.patch.object(
                workflow,
                "resolve_include_spec_dirs",
                return_value=[include_dir],
            ),
            mock.patch.object(workflow, "missing_required_paths", return_value=[]),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
            run_mock.return_value = mock.Mock(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()):
                result = workflow.main([
                    "--include-spec-dir",
                    str(include_dir),
                ])

        self.assertEqual(0, result)
        self.assertIn(
            str(include_dir),
            run_mock.call_args_list[0].args[0],
        )

    def test_missing_linked_solver_specs_are_cached_from_polysolve_pin(self):
        workflow = import_workflow()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            polyfem_source_dir = tmp_dir / "polyfem"
            json_spec_dir = polyfem_source_dir / "json-specs"
            cmake_dir = polyfem_source_dir / "cmake" / "recipes"
            cache_dir = tmp_dir / "build" / "polysolve-json-specs"
            json_spec_dir.mkdir(parents=True)
            cmake_dir.mkdir(parents=True)
            schema_file = json_spec_dir / "input-spec.json"
            schema_file.write_text("{}", encoding="utf-8")
            (cmake_dir / "polysolve.cmake").write_text(
                'CPMAddPackage("gh:polyfem/polysolve#abc123")',
                encoding="utf-8",
            )

            def fake_urlretrieve(url, target):
                Path(target).write_text("{}", encoding="utf-8")
                return target, None

            with (
                mock.patch.object(workflow, "POLYSOLVE_SPEC_CACHE_DIR", cache_dir),
                mock.patch.object(workflow.urllib.request, "urlretrieve", fake_urlretrieve),
            ):
                include_dirs = workflow.resolve_include_spec_dirs(
                    schema_file,
                    [],
                    polyfem_source_dir,
                )

            self.assertEqual([cache_dir], include_dirs)
            self.assertTrue((cache_dir / "linear-solver-spec.json").exists())
            self.assertTrue((cache_dir / "nonlinear-solver-spec.json").exists())

    def test_check_workflow_runs_backend_free_checks_after_generation(self):
        workflow = import_workflow()

        with (
            mock.patch.object(workflow, "resolve_include_spec_dirs", return_value=[]),
            mock.patch.object(workflow, "missing_required_paths", return_value=[]),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
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
                    str(POLYFEM_SCHEMA_FILE),
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
                    str(ROOT / "python-from-jse" / "tools" / "regenerate_and_test.py"),
                ],
            ],
            [call.args[0] for call in run_mock.call_args_list],
        )
        self.assertEqual(
            [ROOT, ROOT, ROOT / "python-from-jse"],
            [call.kwargs["cwd"] for call in run_mock.call_args_list],
        )
        self.assertNotIn("env", run_mock.call_args_list[2].kwargs)

    def test_workflow_reports_missing_required_directory(self):
        workflow = import_workflow()

        def fake_exists(path: Path) -> bool:
            return path != ROOT / "python-from-jse"

        with (
            mock.patch.object(workflow, "resolve_include_spec_dirs", return_value=[]),
            mock.patch.object(workflow.Path, "exists", fake_exists),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = workflow.main([])

        self.assertEqual(1, result)
        self.assertIn("Missing required path", stderr.getvalue())
        self.assertIn("python-from-jse", stderr.getvalue())
        self.assertIn("submodule", stderr.getvalue())
        run_mock.assert_not_called()

    def test_missing_polyfem_schema_points_to_submodule_setup(self):
        workflow = import_workflow()

        def fake_exists(path: Path) -> bool:
            return path != POLYFEM_SCHEMA_FILE

        with (
            mock.patch.object(workflow, "resolve_include_spec_dirs", return_value=[]),
            mock.patch.object(workflow.Path, "exists", fake_exists),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = workflow.main([])

        self.assertEqual(1, result)
        self.assertIn("polyfem", stderr.getvalue())
        self.assertIn("submodule", stderr.getvalue())
        run_mock.assert_not_called()

    def test_missing_linked_solver_spec_points_to_include_spec_dir(self):
        workflow = import_workflow()

        def fake_exists(path: Path) -> bool:
            return path not in {
                POLYFEM_SCHEMA_FILE.parent / "linear-solver-spec.json",
                POLYFEM_SCHEMA_FILE.parent / "nonlinear-solver-spec.json",
            }

        with (
            mock.patch.object(workflow, "resolve_include_spec_dirs", return_value=[]),
            mock.patch.object(workflow.Path, "exists", fake_exists),
            mock.patch.object(workflow.subprocess, "run") as run_mock,
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = workflow.main([])

        self.assertEqual(1, result)
        self.assertIn("linear-solver-spec.json", stderr.getvalue())
        self.assertIn("--include-spec-dir", stderr.getvalue())
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
