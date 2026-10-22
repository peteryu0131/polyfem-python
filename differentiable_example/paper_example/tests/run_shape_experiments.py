"""Optional experiment runner: shared baseline, resume checks and three design modes.

Run this file explicitly; importing shape_opt only builds its configuration.
Use a fresh run directory after changing the example or backend.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import _experiment_records as records
from _lattice_geometry import LatticeGeometry
from polyfempy import differentiable_api as diff
from shape_opt import config, objective, CONTACT_STIFFNESS

RUN_DIR = HERE / "runs" / "shape_comparison_01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["thickness", "angle", "joint", "all"], default="thickness")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    config.output.directory = str(root / "baseline")
    geometry = LatticeGeometry(HERE)
    initial_h, initial_angle = geometry.initial_thickness, geometry.initial_angle
    # Conservative one-update experiments, not a convergence study.
    thickness_step, angle_step = initial_h * .001, .01
    bounds = [(initial_h*.99, initial_h*1.01), (89.9, 90.1)]
    sources = [__file__, HERE / "shape_opt.py", HERE / "_lattice_geometry.py"]
    spec = records.fingerprint(config, sources)
    if root.exists():
        if json.loads((root / "experiment.json").read_text()) != spec:
            raise ValueError("Mesh, source or configuration changed; use a new run directory")
    else:
        root.mkdir(parents=True)
        records.save_json(root / "experiment.json", spec)
        records.save_json(root / "mesh_measurements.json", geometry.report)
    print("Reference mesh:", geometry.report, flush=True)
    geometry.check(initial_h, initial_angle)
    for h in bounds[0]:
        for angle in bounds[1]:
            geometry.check(h, angle)

    thickness = torch.tensor(initial_h, dtype=torch.float64, requires_grad=True)
    angle = torch.tensor(initial_angle, dtype=torch.float64, requires_grad=True)
    baseline_path = root / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    else:
        (root / "baseline").mkdir(exist_ok=False)
        # Verify imported mesh ordering, rather than guessing how bodies are joined.
        from polyfempy import polyfempy as backend
        session = backend.DifferentiableSession()
        session.set_settings(config.as_dict())
        np.testing.assert_allclose(session.mesh_vertices(), geometry.base.numpy(), rtol=0, atol=1e-12)
        del session
        problem = diff.shape_opt(
            diff.model(config), parameters={"thickness": thickness, "angle": angle},
            geometry=geometry, objective=objective,
        )
        print("Baseline: forward and one backward for both design parameters", flush=True)
        value = problem()
        value.backward()
        baseline = {"peak_MPa": float(value.detach()), "thickness_cm": initial_h,
                    "angle_degrees": initial_angle,
                    "gradient_thickness": float(thickness.grad), "gradient_angle": float(angle.grad)}
        if not all(np.isfinite(v) for v in baseline.values()):
            raise ValueError("Non-finite baseline or gradient")
        records.save_json(baseline_path, baseline)
        del value, problem
    print("Baseline:", baseline, flush=True)

    modes = ["thickness", "angle", "joint"] if args.mode == "all" else [args.mode]
    for mode in modes:
        result_path = root / f"{mode}.json"
        if result_path.exists():
            print(mode, json.loads(result_path.read_text()), flush=True)
            continue
        # Each experiment starts from the SAME original geometry and gradient.
        with torch.no_grad():
            thickness.fill_(initial_h)
            angle.fill_(initial_angle)
        thickness.grad = thickness.new_tensor(baseline["gradient_thickness"])
        angle.grad = angle.new_tensor(baseline["gradient_angle"])
        optimizer = torch.optim.Adam([
            {"params": [thickness], "lr": thickness_step if mode != "angle" else 0.},
            {"params": [angle], "lr": angle_step if mode != "thickness" else 0.},
        ])
        optimizer.step()
        h, theta = float(thickness.detach()), float(angle.detach())
        checks = geometry.check(h, theta)
        candidate_dir = root / mode
        candidate_dir.mkdir(exist_ok=False)
        config.output.directory = str(candidate_dir)
        records.save_json(candidate_dir / "config.json", config.as_dict())
        print(f"{mode}: thickness={h:.10g} cm, angle={theta:.10g} deg; candidate forward", flush=True)
        problem = diff.shape_opt(diff.model(config),
            parameters={"thickness": thickness, "angle": angle}, geometry=geometry, objective=objective)
        with torch.no_grad():
            after = float(problem())
        result = {"mode": mode, "before_MPa": baseline["peak_MPa"], "after_MPa": after,
                  "fixed_contact_stiffness": CONTACT_STIFFNESS,
                  "thickness_before_cm": initial_h, "thickness_after_cm": h,
                  "angle_before_degrees": initial_angle, "angle_after_degrees": theta,
                  "relative_change": after/baseline["peak_MPa"]-1,
                  "accepted": after < baseline["peak_MPa"], **checks,
                  "scope": "One gradient update, verified by a new forward; not a converged optimum"}
        records.save_json(result_path, result)
        print(json.dumps(result, indent=2), flush=True)
        del problem


