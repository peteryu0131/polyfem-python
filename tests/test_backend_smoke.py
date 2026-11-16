from pathlib import Path
import json
import os
import re
import subprocess
import sys

import pytest


def _load_expected_error_metrics(config_path: Path) -> tuple[dict[str, float], float]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    tests = payload.get("tests")
    if not isinstance(tests, dict):
        raise AssertionError(f"backend smoke config has no tests block: {config_path}")

    margin = float(tests.get("margin", 1e-8))
    expected = {
        key: float(value)
        for key, value in tests.items()
        if key.startswith("err_") and isinstance(value, (int, float))
    }
    if not expected:
        raise AssertionError(f"backend smoke config has no err_* test metrics: {config_path}")
    return expected, margin


def _load_actual_error_metrics(output_dir: Path) -> dict[str, float]:
    stats_path = output_dir / "sim.json"
    if not stats_path.exists():
        raise AssertionError(f"backend did not write expected stats file: {stats_path}")

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        key: float(value)
        for key, value in payload.items()
        if key.startswith("err_") and isinstance(value, (int, float))
    }


@pytest.fixture
def backend_root():
    root = Path(__file__).resolve().parents[1]

    backend_info = subprocess.run(
        [sys.executable, "-m", "polyfempy", "backend-info", "--require"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if backend_info.returncode != 0:
        pytest.skip(
            "C++ backend is unavailable: "
            + (backend_info.stderr.strip() or backend_info.stdout.strip())
        )

    return root


def test_backend_forward_solve_smoke(tmp_path, backend_root):
    root = backend_root
    config_path = root / "polyfem-data" / "units" / "neohookean.json"
    if not config_path.exists():
        pytest.skip(f"backend smoke config is missing: {config_path}")
    expected_metrics, margin = _load_expected_error_metrics(config_path)

    output_dir = tmp_path / "backend_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "polyfempy",
            "solve",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "2",
            "--max-threads",
            "1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output_dir.exists()
    assert any(output_dir.iterdir())

    actual_metrics = _load_actual_error_metrics(output_dir)
    for key, expected_value in expected_metrics.items():
        assert key in actual_metrics
        assert abs(actual_metrics[key] - expected_value) <= margin


def test_readme_forward_example(tmp_path, backend_root):
    readme = (backend_root / "README.md").read_text(encoding="utf-8")
    snippets = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    assert len(snippets) == 1, "Expected one runnable README example"
    script = tmp_path / "readme_example.py"
    script.write_text(
        snippets[0]
        + "\nimport math\n"
        + "assert math.isfinite(float(abs(result.sol).max()))\n"
        + "assert math.isclose(float(abs(result.sol).max()), 0.00011235, rel_tol=5e-4)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(backend_root / "python"), str(backend_root),
    ])
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
