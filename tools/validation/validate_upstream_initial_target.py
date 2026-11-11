"""Initial-condition target + min-distance checks using the upstream contact case."""

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
    source = ROOT / "differentiability-data/input/initial-contact-min-dist-opt.json"
    spec = json.loads(source.read_text())
    configs = []
    for index, entry in enumerate(spec["states"]):
        path = (source.parent / entry["path"]).resolve()
        config = json.loads(path.read_text())
        for geometry in config["geometry"]:
            geometry["mesh"] = str((path.parent / geometry["mesh"]).resolve())
        config["solver"]["max_threads"] = 1
        config["solver"]["linear"]["solver"] = "Eigen::SimplicialLDLT"
        config["output"] = {
            "directory": str(output / f"state{index}"), "json": "",
            "log": {"level": "warning"}, "paraview": {"file_name": ""},
            "advanced": {"save_time_sequence": False},
        }
        configs.append(config)
    # Upstream target no longer reads this legacy field; its test loads non-strictly.
    assert spec["functionals"][0]["static_objective"].pop("matching") == "exact"
    objectives = [diff.objectives.create(item["type"], **{k: v for k, v in item.items() if k != "type"})
                  for item in spec["functionals"]]
    session = backend.DifferentiableSession()
    session.set_settings(configs[0])
    # Native FE order includes P2 nodes; do not infer it from mesh vertices.
    initial = torch.tensor(session.initial_condition_parameters(), dtype=torch.float64)
    x = initial.reshape(2, -1).T.contiguous().requires_grad_()
    assert x.numel() == spec["parameters"][0]["number"]
    del session
    matching = diff.initial(configs[0], target=configs[1], objective=objectives[0])
    distance = diff.initial(configs[0], objective=objectives[1])

    def values(x):
        return matching(initial_condition=x), distance(initial_condition=x)

    report = {
        "upstream_test": "initial-contact-min-dist", "source": str(source.relative_to(ROOT)),
        "scope": "Python adaptation; original C++ test executable not run",
        "adaptations": ["absolute mesh paths", "one thread", "SimplicialLDLT", "separate output",
                        "PyTorch seed 42 instead of C++ RNG", "two FD steps",
                        "omit obsolete matching=exact (not read by upstream TargetForm)",
                        "sum of two scalar API calls in PyTorch, with separate component checks"],
        "commits": {name: subprocess.check_output(["git", "-C", str(ROOT/name), "rev-parse", "HEAD"], text=True).strip()
                    for name in ("polyfem", "differentiability-data")},
        "shape": list(x.shape), "objectives": [obj.as_dict() for obj in objectives],
        "rtol": 1.9e-6, "atol": 1e-8, "checks": [], "passed": False,
    }
    try:
        a, b = values(x)
        ga, = torch.autograd.grad(a, x)
        gb, = torch.autograd.grad(b, x)
        assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
        report["values"] = {"target": a.item(), "min_dist": b.item(), "sum": (a+b).item()}
        print(json.dumps(report["values"]), flush=True)
        generator = torch.Generator().manual_seed(42)
        for trial in range(3):
            direction = 2*torch.rand(x.shape, generator=generator, dtype=x.dtype)-1
            analytic = [(g*direction).sum().item() for g in (ga, gb, ga+gb)]
            for h in (1e-5, 3e-5):
                with torch.no_grad():
                    ap, bp = values(x+h*direction)
                    am, bm = values(x-h*direction)
                fd = [(ap-am).item()/(2*h), (bp-bm).item()/(2*h), ((ap+bp)-(am+bm)).item()/(2*h)]
                for name, actual, expected in zip(("target", "min_dist", "sum"), analytic, fd):
                    error = abs(actual-expected)
                    check = {"trial": trial, "h": h, "component": name,
                             "autograd": actual, "finite_difference": expected,
                             "relative_error": error/max(abs(actual), abs(expected), 1e-30),
                             "passed": error <= report["atol"]+report["rtol"]*abs(expected)}
                    report["checks"].append(check)
                    print(json.dumps(check), flush=True)
        parameter = torch.zeros((), dtype=x.dtype, requires_grad=True)
        ma, mb = values(x.detach()+parameter*direction)
        mapped, = torch.autograd.grad((ma+mb).square(), parameter)
        expected = 2*(a+b).item()*((ga+gb)*direction).sum().item()
        report["mapping"] = {"autograd": mapped.item(), "expected": expected,
                             "passed": abs(mapped.item()-expected) <= 1e-8+1e-10*abs(expected)}
        report["passed"] = all(item["passed"] for item in report["checks"]) and report["mapping"]["passed"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        (output/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not report["passed"]:
        raise AssertionError("Initial target gradient check failed; see result.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    validate(parser.parse_args().output)
