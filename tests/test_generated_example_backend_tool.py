from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "check_generated_example_backend.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "check_generated_example_backend_for_test",
        TOOL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def test_generated_example_backend_tool_exists_and_documents_manual_use():
    text = TOOL_PATH.read_text(encoding="utf-8")

    assert "generated example" in text
    assert "--example" in text
    assert "--source-json" in text
    assert "--require-tests-match" in text


def test_compare_metric_maps_reports_pass_and_fail_rows():
    tool = _load_tool_module()

    rows = tool.compare_metric_maps(
        left={"err_l2": 1.0, "err_h1": 2.0},
        right={"err_l2": 1.0 + 1e-12, "err_h1": 2.1},
        tolerance=1e-9,
    )

    by_key = {row.key: row for row in rows}
    assert by_key["err_l2"].passed
    assert not by_key["err_h1"].passed


def test_backend_workflow_exposes_optional_generated_example_diagnostic():
    workflow = (ROOT / ".github" / "workflows" / "backend.yml").read_text(
        encoding="utf-8"
    )

    assert "run_generated_example_diagnostic" in workflow
    assert "tools/check_generated_example_backend.py" in workflow
    assert "contact_2d_golf_ball_deformable_wall_generated_api.py" in workflow
