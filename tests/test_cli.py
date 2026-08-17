from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_backend_info_reports_backend_status(capsys):
    from polyfempy.cli import main

    exit_code = main(["backend-info"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "backend_available=" in captured.out


def test_backend_info_can_require_backend(monkeypatch, capsys):
    from polyfempy import cli

    fake_polyfempy = types.SimpleNamespace(
        cpp_backend_available=lambda: False,
        cpp_backend_error=lambda: RuntimeError("not built"),
    )
    monkeypatch.setattr(cli, "_load_polyfempy", lambda: fake_polyfempy)

    exit_code = cli.main(["backend-info", "--require"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "C++ backend not loaded" in captured.err


def test_solve_reports_missing_backend_cleanly(monkeypatch, capsys, tmp_path):
    from polyfempy import cli

    fake_polyfempy = types.SimpleNamespace(
        cpp_backend_available=lambda: False,
        cpp_backend_error=lambda: RuntimeError("not built"),
    )
    monkeypatch.setattr(cli, "_load_polyfempy", lambda: fake_polyfempy)
    config = tmp_path / "input.json"
    config.write_text("{}", encoding="utf-8")

    exit_code = cli.main(["solve", str(config)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "C++ backend not loaded" in captured.err


def test_solve_dispatches_json_config_to_backend(monkeypatch, tmp_path):
    from polyfempy import cli

    calls = []

    def polyfem_command(**kwargs):
        calls.append(kwargs)

    fake_polyfempy = types.SimpleNamespace(
        cpp_backend_available=lambda: True,
        cpp_backend_error=lambda: None,
        polyfem_command=polyfem_command,
    )
    monkeypatch.setattr(cli, "_load_polyfempy", lambda: fake_polyfempy)

    config = tmp_path / "input.json"
    config.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        [
            "solve",
            str(config),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "4",
            "--max-threads",
            "2",
            "--no-strict-validation",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "json": str(config),
            "yaml": "",
            "log_level": 4,
            "strict_validation": False,
            "max_threads": 2,
            "output_dir": str(output_dir),
        }
    ]


def test_solve_dispatches_yaml_config_to_backend(monkeypatch, tmp_path):
    from polyfempy import cli

    calls = []

    def polyfem_command(**kwargs):
        calls.append(kwargs)

    fake_polyfempy = types.SimpleNamespace(
        cpp_backend_available=lambda: True,
        cpp_backend_error=lambda: None,
        polyfem_command=polyfem_command,
    )
    monkeypatch.setattr(cli, "_load_polyfempy", lambda: fake_polyfempy)

    config = tmp_path / "input.yaml"
    config.write_text("problem: test\n", encoding="utf-8")

    exit_code = cli.main(["solve", str(config)])

    assert exit_code == 0
    assert calls == [
        {
            "json": "",
            "yaml": str(config),
            "log_level": 2,
            "strict_validation": True,
            "max_threads": 1,
            "output_dir": "",
        }
    ]


def test_python_module_help_works_without_backend():
    result = subprocess.run(
        [sys.executable, "-m", "polyfempy", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "backend-info" in result.stdout
    assert "solve" in result.stdout


def test_pyproject_declares_console_script():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'polyfempy = "polyfempy.cli:main"' in pyproject
    assert "cmake>=3.25" in pyproject


def test_setup_py_keeps_console_script_for_legacy_build_metadata():
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "console_scripts" in setup_py
    assert "polyfempy=polyfempy.cli:main" in setup_py


def test_setup_py_always_passes_conda_prefix_to_cmake():
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")

    conda_prefix_line = next(
        line for line in setup_py.splitlines() if "CONDA_PREFIX" in line
    )
    assert conda_prefix_line.startswith("        conda_prefix = ")


def test_setup_py_disables_heavy_optional_polysolve_dependencies():
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "'-DPOLYSOLVE_WITH_SPECTRA=OFF'" in setup_py
    assert "'-DPOLYSOLVE_WITH_AMGCL=OFF'" in setup_py
    assert "'-DPOLYSOLVE_WITH_UMFPACK=OFF'" in setup_py
    assert "'-DPOLYSOLVE_WITH_HYPRE=OFF'" in setup_py


def test_github_actions_generates_ignored_api_before_tests():
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )

    assert "matrix:" in workflow
    assert '"3.10"' in workflow
    assert '"3.11"' in workflow
    assert '"3.12"' in workflow
    assert "submodules: recursive" in workflow
    assert "python tools/generate_polyfem_api.py" in workflow
    assert "python -m pytest tests -q" in workflow


def test_backend_github_actions_builds_compiled_backend_manually():
    workflow = (ROOT / ".github" / "workflows" / "backend.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "submodules: recursive" in workflow
    assert "python tools/generate_polyfem_api.py" in workflow
    assert "python -m pip install -e . --no-build-isolation -vv" in workflow
    assert "python -m polyfempy backend-info --require" in workflow
    assert "python -m pytest tests/test_backend_smoke.py -q -rs" in workflow


def test_generated_contact_backend_workflow_runs_manual_full_sweep():
    workflow = (
        ROOT / ".github" / "workflows" / "generated-contact-backend.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "submodules: recursive" in workflow
    assert "python tools/generate_polyfem_api.py" in workflow
    assert "python -m pip install -e . --no-build-isolation -vv" in workflow
    assert "python -m polyfempy backend-info --require" in workflow
    assert "tools/run_generated_contact_backend_checks.py" in workflow
    assert "--require-tests-match" in workflow
    assert "tools/generated_contact_expected_failures.json" in workflow
    assert "actions/upload-artifact@v4" in workflow
