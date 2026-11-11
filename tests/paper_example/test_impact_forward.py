from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET

import numpy as np

import pytest


PAPER_EXAMPLE = Path(__file__).resolve().parents[2] / "differentiable_example" / "paper_example"

SCRIPT = PAPER_EXAMPLE / "impact_forward.py"


def load_example():
    tree = ast.parse(SCRIPT.read_text())
    # Read the configuration without starting the top-level simulation.
    split = next(
        i + 1 for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "config"
                for target in node.targets)
    )
    example = {"__file__": str(SCRIPT), "__name__": "impact_forward_test"}
    exec(compile(ast.Module(body=tree.body[:split], type_ignores=[]), str(SCRIPT), "exec"), example)
    example["execution_body"] = tree.body[split:]
    return example


def run_example(example, output_dir):
    example["OUTPUT_DIR"] = output_dir
    example["config"].output.directory = str(output_dir)
    exec(compile(ast.Module(body=example["execution_body"], type_ignores=[]),
                 str(SCRIPT), "exec"), example)
    return example["result"]


def verify_outputs(output_dir, *, dt, end_time):
    """Check this experiment's exported samples, not a differentiable objective."""
    import meshio

    entries = ET.parse(output_dir / "impact.pvd").findall("./Collection/DataSet")
    expected_times = np.arange(round(end_time / dt) + 1) * dt
    times = np.array([float(entry.attrib["timestep"]) for entry in entries])
    if len(times) != len(expected_times) or not np.allclose(times, expected_times):
        raise RuntimeError("Incomplete exported time sequence")
    history = []
    for entry, time in zip(entries, times):
        blocks = ET.parse(output_dir / entry.attrib["file"])
        volume_file = blocks.find(".//Block[@name='Volume']/DataSet").attrib["file"]
        contact_file = blocks.find(".//Block[@name='Contact']/DataSet").attrib["file"]
        mesh = meshio.read(output_dir / volume_file)
        contact = meshio.read(output_dir / contact_file)
        fields = mesh.point_data
        for key in ("body_ids", "displacement", "von_mises", "cauchy_stess_1", "cauchy_stess_2"):
            if key not in fields or not np.isfinite(fields[key]).all():
                raise RuntimeError(f"Missing or non-finite output field: {key}")
        body_ids = fields["body_ids"].ravel()
        lattice, block = body_ids == 1, body_ids == 2
        if not lattice.any() or not block.any():
            raise RuntimeError("Both bodies must appear in the volume output")
        displacement = fields["displacement"]
        bottom = lattice & (np.abs(mesh.points[:, 1]) < 1e-8)
        if not bottom.any() or np.max(np.abs(displacement[bottom])) > 1e-8:
            raise RuntimeError("Lattice bottom is not fixed")
        force = contact.point_data["contact_forces"]
        if not np.isfinite(force).all():
            raise RuntimeError("Non-finite contact forces")
        # VTU points are reference coordinates. The original block is wholly
        # above the lattice, so this separates its collision vertices exactly.
        block_surface = contact.points[:, 1] >= mesh.points[block, 1].min() - 1e-8
        triangles = mesh.cells_dict["triangle"]
        rest = mesh.points[triangles, :2]
        moved = (mesh.points + displacement)[triangles, :2]
        def signed_area2(points):
            a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        ratio = signed_area2(moved) / signed_area2(rest)
        if not np.isfinite(ratio).all() or ratio.min() <= 0:
            raise RuntimeError("Inverted or degenerate deformed triangle")
        history.append({
            "time_s": float(time),
            "block_mean_displacement_y_cm": float(displacement[block, 1].mean()),
            "block_contact_force_y_g_cm_s2": float(force[block_surface, 1].sum()),
            "fixed_bottom_max_displacement_cm": float(np.abs(displacement[bottom]).max()),
            # g/(cm*s^2) = 0.1 Pa; report visualization-sampled maxima, not
            # a continuous/global peak or a differentiable stress objective.
            "lattice_sampled_max_von_mises_MPa": float(fields["von_mises"][lattice].max() / 1e7),
            "min_area_ratio": float(ratio.min()),
        })
    if history[-1]["block_mean_displacement_y_cm"] >= -0.01:
        raise RuntimeError("Block did not fall; check imported boundary labels")
    contact_steps = [row for row in history if row["block_contact_force_y_g_cm_s2"] > 1e-6]
    if not contact_steps or history[-1]["lattice_sampled_max_von_mises_MPa"] <= 0:
        raise RuntimeError("Impact contact and nonzero lattice stress were not observed")
    summary = {"status": "verified", "dt_s": dt, "end_time_s": end_time,
               "first_sample_with_block_contact_s": contact_steps[0]["time_s"], "history": history}
    (output_dir / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def test_impact_config_keeps_block_free_and_preserves_units(tmp_path):
    example = load_example()
    config = example["config"].as_dict()
    assert config["units"] == {"length": "cm", "mass": "g", "time": "s"}
    assert [m["E"]["unit"] for m in config["materials"]] == ["MPa", "GPa"]
    assert config["time"]["dt"] == 0.01
    assert config["time"]["tend"] == 0.5
    geometry = config["geometry"]
    assert all(Path(item["mesh"]).is_file() for item in geometry)
    assert geometry[0]["surface_selection"][0]["id"] == 3
    assert geometry[1]["surface_selection"] == 4
    assert [bc["id"] for bc in config["boundary_conditions"]["dirichlet_boundary"]] == [3]
    assert config["contact"]["enabled"]
    assert config["output"]["paraview"]["options"]["contact_forces"]


def test_real_lattice_impact_contact_and_stress(tmp_path, monkeypatch):
    pytest.importorskip("polyfempy.polyfempy")
    pytest.importorskip("meshio")
    example = load_example()
    # Keep this regression short and resolve the pre-contact free-fall steps.
    monkeypatch.setattr(example["config"], "time",
                        example["polyfem"].time(example["polyfem"].object1(
                            t0=0, dt=0.005, tend=0.03, integrator="BDF")))
    result = run_example(example, tmp_path)
    assert np.isfinite(result.sol).all()
    summary = verify_outputs(tmp_path, dt=0.005, end_time=0.03)
    rows = summary["history"]
    assert summary["status"] == "verified"
    assert len(rows) == 7
    # Before impact, BDF1 free-fall is -g*dt^2*n*(n+1)/2.
    for step in (1, 2):
        assert rows[step]["block_mean_displacement_y_cm"] == pytest.approx(
            -980 * summary["dt_s"] ** 2 * step * (step + 1) / 2, abs=1e-6)
        assert rows[step]["block_contact_force_y_g_cm_s2"] == pytest.approx(0, abs=1e-6)
    assert rows[-1]["block_mean_displacement_y_cm"] > -980 * summary["dt_s"] ** 2 * 6 * 7 / 2
    assert rows[-1]["lattice_sampled_max_von_mises_MPa"] > rows[2]["lattice_sampled_max_von_mises_MPa"]


def test_script_saves_config_and_solves(tmp_path, capsys):
    example = load_example()
    calls = []
    result = object()

    def solve(*, cfg):
        saved = json.loads((tmp_path / "config.json").read_text())
        assert saved == cfg.as_dict()
        calls.append(cfg)
        return result

    example["solve"] = solve
    assert run_example(example, tmp_path) is result
    assert calls == [example["config"]]
    assert calls[0].as_dict()["output"]["directory"] == str(tmp_path)
    assert str(tmp_path / "impact.pvd") in capsys.readouterr().out
