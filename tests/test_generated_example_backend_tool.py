from __future__ import annotations

import json
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


def test_backend_tool_applies_polyfem_test_time_steps_to_tend_dt_config(tmp_path):
    tool = _load_tool_module()

    source_json = tmp_path / "example.json"
    source_json.write_text(
        json.dumps(
            {
                "time": {"tend": 2.0, "dt": 0.25},
                "tests": {"time_steps": 3},
            }
        ),
        encoding="utf-8",
    )

    payload, test_directive = tool.load_reduced_backend_payload(source_json)

    assert test_directive == 3
    assert payload["time"] == {"dt": 0.25, "time_steps": 3}


def test_backend_tool_applies_polyfem_test_time_steps_to_tend_steps_config(tmp_path):
    tool = _load_tool_module()

    source_json = tmp_path / "example.json"
    source_json.write_text(
        json.dumps(
            {
                "time": {"tend": 2.0, "time_steps": 8},
                "tests": {"time_steps": 2},
            }
        ),
        encoding="utf-8",
    )

    payload, test_directive = tool.load_reduced_backend_payload(source_json)

    assert test_directive == 2
    assert payload["time"] == {"dt": 0.25, "time_steps": 2}


def test_backend_tool_defaults_dynamic_examples_to_one_step(tmp_path):
    tool = _load_tool_module()

    source_json = tmp_path / "example.json"
    source_json.write_text(
        json.dumps({"time": {"dt": 0.1, "time_steps": 10}}),
        encoding="utf-8",
    )

    payload, test_directive = tool.load_reduced_backend_payload(source_json)

    assert test_directive == 1
    assert payload["time"] == {"dt": 0.1, "time_steps": 1}


def test_backend_tool_applies_polyfem_verify_run_linear_solver_patch(tmp_path):
    tool = _load_tool_module()

    default_source = tmp_path / "example.json"
    default_source.write_text(
        json.dumps(
            {
                "solver": {"linear": {"solver": ["A", "B"]}},
                "time": {"dt": 0.1, "time_steps": 10},
            }
        ),
        encoding="utf-8",
    )
    navier_source = tmp_path / "navier-example.json"
    navier_source.write_text(
        json.dumps(
            {
                "solver": {"linear": {"solver": ["A", "B"]}},
                "time": {"dt": 0.1, "time_steps": 10},
            }
        ),
        encoding="utf-8",
    )

    default_payload, _ = tool.load_reduced_backend_payload(default_source)
    navier_payload, _ = tool.load_reduced_backend_payload(navier_source)

    assert (
        default_payload["solver"]["linear"]["solver"]
        == "Eigen::SimplicialLDLT"
    )
    assert navier_payload["solver"]["linear"]["solver"] == "Eigen::SparseLU"


def test_backend_tool_restores_generated_payload_semantics_before_backend_run(tmp_path):
    tool = _load_tool_module()

    class GeneratedConfig:
        def as_dict(self):
            return {
                "geometry": [
                    {
                        "mesh": str(tmp_path / "mesh.obj"),
                        "transformation": {
                            "rotation": [90.0],
                            "scale": [0.5],
                            "translation": [],
                        },
                    }
                ],
            }

    payload = tool._config_payload_for_backend_run(GeneratedConfig())

    assert payload["geometry"][0]["transformation"] == {
        "rotation": 90.0,
        "scale": 0.5,
    }


def test_backend_tool_keeps_all_and_static_test_directives_unpatched(tmp_path):
    tool = _load_tool_module()

    all_source = tmp_path / "all.json"
    all_source.write_text(
        json.dumps(
            {
                "time": {"dt": 0.1, "time_steps": 10},
                "tests": {"time_steps": "all"},
            }
        ),
        encoding="utf-8",
    )
    static_source = tmp_path / "static.json"
    static_source.write_text(json.dumps({"tests": {}}), encoding="utf-8")

    all_payload, all_directive = tool.load_reduced_backend_payload(all_source)
    static_payload, static_directive = tool.load_reduced_backend_payload(static_source)

    assert all_directive == "all"
    assert all_payload["time"] == {"dt": 0.1, "time_steps": 10}
    assert static_directive == "static"
    assert "time" not in static_payload


def test_backend_tool_prunes_visual_output_files(tmp_path):
    tool = _load_tool_module()

    keep = tmp_path / "sim.json"
    keep.write_text("{}", encoding="utf-8")
    log = tmp_path / "polyfem.log"
    log.write_text("log", encoding="utf-8")
    for suffix in (".vtu", ".vtm", ".pvd"):
        (tmp_path / f"sim{suffix}").write_text("large", encoding="utf-8")

    removed = tool.prune_visual_outputs(tmp_path)

    assert sorted(path.suffix for path in removed) == [".pvd", ".vtm", ".vtu"]
    assert keep.exists()
    assert log.exists()
    assert not any(tmp_path.glob("*.vtu"))
    assert not any(tmp_path.glob("*.vtm"))
    assert not any(tmp_path.glob("*.pvd"))


def test_backend_workflow_exposes_optional_generated_example_diagnostic():
    workflow = (ROOT / ".github" / "workflows" / "backend.yml").read_text(
        encoding="utf-8"
    )

    assert "run_generated_example_diagnostic" in workflow
    assert "tools/check_generated_example_backend.py" in workflow
    assert "contact_2d_golf_ball_deformable_wall_generated_api.py" in workflow
