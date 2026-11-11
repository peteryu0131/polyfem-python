"""Check the callable shape API using PolyFEM's shape-transient-friction case.

This ports the directional derivative check, not the C++ test executable.
Run: python -m tools.validation.validate_upstream_transient --output PATH
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
    source = ROOT / "differentiability-data/input/shape-transient-friction-opt.json"
    spec = json.loads(source.read_text())
    state = source.parent / spec["states"][0]["path"]
    config = json.loads(state.read_text())
    for geometry in config["geometry"]:
        geometry["mesh"] = str((state.parent / geometry["mesh"]).resolve())
    config["solver"]["max_threads"] = 1
    config["solver"]["linear"]["solver"] = "Eigen::SimplicialLDLT"
    original_grad_norm_tol = config["solver"]["nonlinear"]["grad_norm_tol"]
    # Match test_diff.cpp: finite differences amplify forward-solve error.
    config["solver"]["nonlinear"]["grad_norm_tol"] = 1e-11
    config["output"] = {
        "directory": str(output), "json": "",
        "log": {"level": "warning"},
        "paraview": {"file_name": ""},
        "advanced": {"save_time_sequence": False},
    }
    objective = diff.objectives.create(
        "transient_integral", state=0, integral_type="uniform",
        static_objective=diff.objectives.stress_norm(),
    )
    source_objective = spec["functionals"][0]
    assert objective.as_dict()["integral_type"] == source_objective["integral_type"]
    assert objective.as_dict()["static_objective"]["type"] == "stress_norm"
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    del session
    obj = diff.shape(config, objective=objective)
    x = vertices.clone().requires_grad_()

    def loss(x):
        return obj(V=x)

    report = {
        "upstream_test": "shape-transient-friction",
        "upstream_file": "polyfem/tests/test_diff.cpp",
        "source": str(source.relative_to(ROOT)),
        "scope": "Python adaptation of upstream directional gradient test; C++ test not executed",
        "objective": objective.as_dict(),
        "parameter_shape": list(x.shape),
        "adaptations": ["absolute mesh paths", "one thread", "SimplicialLDLT",
                        "separate output", "PyTorch RNG instead of C++ mt19937_64",
                        "two finite-difference steps instead of one"],
        "forward_solver_tolerance": {
            "json_grad_norm_tol": original_grad_norm_tol,
            "used_grad_norm_tol": config["solver"]["nonlinear"]["grad_norm_tol"],
            "reason": "Match upstream shape-transient-friction unit test override",
        },
        "seed": 42,
        "rtol": 1.1e-5,
        "atol": 1e-8,
        "upstream_tolerance": "Catch Approx epsilon=1.1e-5; no explicit absolute margin",
        "commits": {name: subprocess.check_output(
            ["git", "-C", str(ROOT / name), "rev-parse", "HEAD"], text=True).strip()
            for name in ("polyfem", "differentiability-data")},
        "checks": [],
        "passed": False,
    }
    try:
        print("Starting transient forward and backward", flush=True)
        value = loss(x)
        gradient, = torch.autograd.grad(value, x)
        assert torch.isfinite(value) and torch.isfinite(gradient).all()
        report["value"] = value.item()
        report["gradient_norm"] = gradient.norm().item()
        print(json.dumps({"value": report["value"], "gradient_norm": report["gradient_norm"]}), flush=True)
        generator = torch.Generator().manual_seed(report["seed"])
        for trial in range(3):
            direction = torch.rand(x.shape, generator=generator, dtype=x.dtype)
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
        # A user-defined scalar controls a full vertex field in ordinary PyTorch.
        parameter = torch.zeros((), dtype=x.dtype, requires_grad=True)
        mapped_value = obj(V=vertices + parameter * direction)
        mapped_gradient, = torch.autograd.grad(mapped_value, parameter)
        expected = (gradient * direction).sum().item()
        chain_passed = abs(mapped_gradient.item() - expected) <= 1e-8 + 1e-10 * abs(expected)
        report["parameter_mapping"] = {
            "expression": "V = original_vertices + parameter * direction",
            "autograd": mapped_gradient.item(), "expected_vjp": expected,
            "passed": chain_passed,
        }
        print(json.dumps(report["parameter_mapping"]), flush=True)
        report["passed"] = chain_passed and all(check["passed"] for check in report["checks"])
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
