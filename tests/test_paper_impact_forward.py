from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "differentiable_example/paper_example/impact_forward.py"


def test_impact_config_keeps_block_free_and_preserves_units(tmp_path):
    example = runpy.run_path(str(SCRIPT))
    config = example["build_config"](tmp_path).as_dict()
    assert config["units"] == {"length": "cm", "mass": "g", "time": "s"}
    assert [m["E"]["unit"] for m in config["materials"]] == ["MPa", "GPa"]
    geometry = config["geometry"]
    assert all(Path(item["mesh"]).is_file() for item in geometry)
    assert geometry[0]["surface_selection"][0]["id"] == 3
    assert geometry[1]["surface_selection"] == 4
    assert [bc["id"] for bc in config["boundary_conditions"]["dirichlet_boundary"]] == [3]
    assert config["contact"]["enabled"]
    assert config["output"]["paraview"]["options"]["contact_forces"]


def test_real_lattice_impact_contact_and_stress(tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("meshio")
    example = runpy.run_path(str(SCRIPT))
    summary = example["main"](tmp_path)
    rows = summary["history"]
    assert summary["status"] == "verified"
    assert len(rows) == 7
    # Before impact, BDF1 free-fall is -g*dt^2*n*(n+1)/2.
    for step in (1, 2):
        assert rows[step]["block_mean_displacement_y_cm"] == pytest.approx(
            -980 * example["DT"] ** 2 * step * (step + 1) / 2, abs=1e-6)
        assert rows[step]["block_contact_force_y_g_cm_s2"] == pytest.approx(0, abs=1e-6)
    assert rows[-1]["block_mean_displacement_y_cm"] > -980 * example["DT"] ** 2 * 6 * 7 / 2
    assert rows[-1]["lattice_sampled_max_von_mises_MPa"] > rows[2]["lattice_sampled_max_von_mises_MPa"]
