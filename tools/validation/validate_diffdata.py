"""Validate original diff-data meshes and objectives; not a convergence study.

Run: python -m tools.validation.validate_diffdata material --output PATH
     python -m tools.validation.validate_diffdata shape --output PATH
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import polyfem.diff as diff
from polyfempy import polyfempy as backend


ROOT = Path(__file__).resolve().parents[2]


def load_case(kind, output):
    name = "topology-compliance" if kind == "material" else "neohookean-stress-3d"
    source = ROOT / "differentiability-data/input" / f"{name}-opt.json"
    spec = json.loads(source.read_text())
    state_path = source.parent / spec["states"][0]["path"]
    config = json.loads(state_path.read_text())
    for geometry in config["geometry"]:
        geometry["mesh"] = str((state_path.parent / geometry["mesh"]).resolve())
    # Only execution/output settings differ from the source state.
    config["solver"]["max_threads"] = 1
    config["solver"]["linear"]["solver"] = "Eigen::SimplicialLDLT"
    config["output"] = {
        "directory": str(output), "json": "",
        "log": {"level": "warning"}, "paraview": {"file_name": ""},
        "advanced": {"save_time_sequence": False},
    }
    return source, spec, config


def validate(kind, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source, spec, config = load_case(kind, output)
    # The raw optimization JSON omits fields supplied by objective-spec defaults.
    source_objective = spec["functionals"][0]
    objective = (diff.objectives.compliance(selection=1) if kind == "material"
                 else diff.objectives.stress_norm(selection=[])).as_dict()
    assert all(objective[key] == value for key, value in source_objective.items())
    # Direct bridge baseline uses original material properties and mesh.
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    session.set_objective(objective)
    session.set_shape_vertices(vertices.numpy(), selection="all")
    baseline = session.solve_objective()
    del session

    if kind == "material":
        obj = diff.material(config, body=1, objective=objective)
        assert obj.n_elements == spec["parameters"][0]["number"] == 198
        assert spec["variable_to_simulation"][0]["composition"] == [
            {"type": "power", "power": 5},
            {"type": "append-const", "size": 198, "value": .3},
            {"type": "E-nu-to-lambda-mu", "is_volume": False},
        ]
        # The source has an empty initial array. Choose x^5 = config E.
        x = torch.full((obj.n_elements,), config["materials"]["E"] ** .2,
                       dtype=torch.float64, requires_grad=True)

        def loss(x):
            return obj(E=x.pow(5), nu=.3)

        # Smaller h amplified numerical noise in the beam's forward differences.
        # Check three larger steps without changing the gradient tolerance.
        lr, steps = .001, (.1, .03, .01)
    else:
        assert spec["parameters"] == "auto"
        assert spec["variable_to_simulation"][0]["composition"] == []
        obj = diff.shape(config, objective=objective)
        x = vertices.clone().requires_grad_()

        def loss(x):
            return obj(V=x)

        lr, steps = 1e-6, (1e-5, 3e-6)

    value = loss(x)
    value.backward()
    initial = float(value.detach())
    np.testing.assert_allclose(initial, baseline, rtol=1e-7, atol=1e-9)
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    generator = torch.Generator().manual_seed(42)
    directions = [x.grad.detach() / x.grad.norm(),
                  torch.randn(x.shape, dtype=x.dtype, generator=generator)]
    checks = []
    for direction in directions:
        direction = direction / direction.norm()
        analytic = float((x.grad * direction).sum())
        for h in steps:
            with torch.no_grad():
                fd = float((loss(x + h * direction) - loss(x - h * direction)) / (2*h))
            error = abs(analytic - fd) / max(abs(analytic), abs(fd), 1e-12)
            checks.append({"h": h, "autograd": analytic, "finite_difference": fd,
                           "relative_error": error})
            print(json.dumps(checks[-1]), flush=True)

    report = {"case": kind, "source": str(source.relative_to(ROOT)),
              "objective": objective, "baseline": baseline, "initial": initial,
              "parameter_shape": list(x.shape), "gradient_checks": checks,
              "baseline_kind": "direct C++ bridge, not upstream optimizer executable",
              "adaptations": ["absolute mesh paths", "one thread", "SimplicialLDLT", "no visualization output"],
              "initialization": "x = E_config ** (1/5); source initial array empty" if kind == "material" else "original loaded vertices"}
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Do not optimize a case whose gradient check failed.
    for check in checks:
        np.testing.assert_allclose(check["autograd"], check["finite_difference"], rtol=3e-3, atol=1e-7)
    history = []
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(3):
        optimizer.zero_grad()
        value = loss(x)
        history.append(float(value.detach()))
        value.backward()
        optimizer.step()
    # A new object independently checks the final design.
    fresh = (diff.material(config, body=1, objective=objective) if kind == "material"
             else diff.shape(config, objective=objective))
    with torch.no_grad():
        final = float(fresh(E=x.pow(5), nu=.3) if kind == "material" else fresh(V=x))
    report.update(history=history, final=final, lr=lr, updates=3,
                  elapsed_seconds=time.perf_counter()-start)
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert final < initial, report
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["material", "shape"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate(args.case, args.output)
