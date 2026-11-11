"""Check the callable shape API using PolyFEM's linear_elasticity-surface case.

This ports the directional derivative check, not the C++ test executable.
Run: python -m tools.validation.validate_upstream_surface --output PATH
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import polyfem.diff as diff
from polyfempy import polyfempy as backend


ROOT = Path(__file__).resolve().parents[2]


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = ROOT / "differentiability-data/input/linear_elasticity-surface-opt.json"
    spec = json.loads(source.read_text())
    state = source.parent / spec["states"][0]["path"]
    config = json.loads(state.read_text())
    for geometry in config["geometry"]:
        geometry["mesh"] = str((state.parent / geometry["mesh"]).resolve())
    config["solver"]["max_threads"] = 1
    config["solver"]["linear"]["solver"] = "Eigen::SimplicialLDLT"
    config["output"] = {
        "directory": str(output), "json": "",
        "log": {"level": "warning"},
        "paraview": {"file_name": ""},
        "advanced": {"save_time_sequence": False},
    }
    objective = diff.objectives.create(**{
        "type_name": "position", "dim": 1, "state": 0, "volume_selection": [],
    })
    assert all(objective.as_dict()[key] == value
               for key, value in spec["functionals"][0].items())
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    del session
    obj = diff.shape(config, objective=objective)
    x = vertices.clone().requires_grad_()

    def loss(x):
        return obj(V=x)

    report = {
        "upstream_test": "linear_elasticity-surface",
        "upstream_file": "polyfem/tests/test_diff.cpp",
        "source": str(source.relative_to(ROOT)),
        "scope": "Python adaptation of upstream directional gradient test; C++ test not executed",
        "objective": objective.as_dict(),
        "parameter_shape": list(x.shape),
        "adaptations": ["absolute mesh paths", "one thread", "SimplicialLDLT",
                        "separate output", "PyTorch RNG instead of C++ mt19937_64",
                        "two finite-difference steps instead of one"],
        "seed": 42,
        "rtol": 2.3e-7,
        "atol": 1e-8,
        "upstream_tolerance": "Catch Approx epsilon=2.3e-7; no explicit absolute margin",
        "commits": {name: subprocess.check_output(
            ["git", "-C", str(ROOT / name), "rev-parse", "HEAD"], text=True).strip()
            for name in ("polyfem", "differentiability-data")},
        "checks": [],
        "passed": False,
    }
    try:
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        assert torch.isfinite(value) and torch.isfinite(gradient).all()
        report["value"] = value.item()
        report["gradient_norm"] = gradient.norm().item()
        generator = torch.Generator().manual_seed(report["seed"])
        for trial in range(3):
            direction = 2 * torch.rand(x.shape, generator=generator, dtype=x.dtype) - 1
            analytic = (gradient * direction).sum().item()
            for h in (1e-6, 3e-6):
                with torch.no_grad():
                    finite_difference = ((loss(x + h * direction) - loss(x - h * direction)) / (2*h)).item()
                error = abs(analytic - finite_difference)
                passed = error <= report["atol"] + report["rtol"] * abs(finite_difference)
                check = {"trial": trial, "h": h, "autograd": analytic,
                         "finite_difference": finite_difference,
                         "relative_error": error / max(abs(analytic), abs(finite_difference), 1e-30),
                         "passed": passed}
                report["checks"].append(check)
                print(json.dumps(check), flush=True)
        report["passed"] = all(check["passed"] for check in report["checks"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not report["passed"]:
        raise AssertionError("Directional gradient check failed; see result.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    validate(parser.parse_args().output)
