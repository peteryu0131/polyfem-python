"""Compare two optimizations/*/run.json problems with the original OptState at the same designs."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import polyfem.diff as diff

from differentiable_example.optimizations.material_and_contact.topology_opt import density_filter
from .compare_original_json import ROOT, run_native, write_json
from .optimization_reference import load, material_loss


def topology_loss(spec, states):
    (mapping,) = spec["variable_to_simulation"]
    assert mapping["state"] == [0, 1] and states[0]["geometry"] == states[1]["geometry"]
    filt, power, scale, nu, convert = mapping["composition"]
    assert [m["type"] for m in mapping["composition"]] == [
        "linear-filter",
        "power",
        "scale",
        "append-const",
        "E-nu-to-lambda-mu",
    ]
    assert convert["is_volume"] is False and filt["state"] == 0
    weights = density_filter(states[0]["geometry"], filt["radius"])
    objects = []
    for f in spec["functionals"]:
        assert f["type"] == "compliance"
        obj = diff.material(
            states[f["state"]], body=1, objective=diff.Objective.COMPLIANCE
        )
        assert obj.n_elements == weights.shape[0] == nu["size"]
        objects.append(obj)

    def filtered_loss(filtered):
        modulus = scale["value"] * filtered.pow(power["power"])
        return torch.stack([obj(E=modulus, nu=nu["value"]) for obj in objects]).sum()

    return weights, filtered_loss


def stage(case, output):
    spec, states = load(case, output)
    for i, (entry, cfg) in enumerate(zip(spec["states"], states, strict=True)):
        staged = output / f"state{i}.json"
        write_json(staged, cfg)
        entry["path"] = str(staged)
    spec["solver"]["max_threads"] = 1
    spec["output"] = {"directory": str(output / "native")}
    write_json(output / "opt.json", spec)
    return spec, states


def validate(case, output):
    spec, states = stage(case, output)
    run_native(
        "polyfem_reference_probe.exe",
        [output / "opt.json", output / "native.json", "--no-strict-validation"],
        output / "native.log",
    )
    ref = json.loads((output / "native.json").read_text())
    initial = spec["parameters"][0]["initial"]
    x = torch.tensor(ref["x"], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(x, x.new_tensor(initial).expand_as(x))
    diagnostics = {}
    if case == "material-opt":
        loss = material_loss(spec, states)
        value = loss(x)
        (gradient,) = torch.autograd.grad(value, x)
    else:
        weights, filtered_loss = topology_loss(spec, states)

        def loss(x):
            return filtered_loss(weights @ x)

        filtered = weights @ x
        value = filtered_loss(filtered)
        gradient, filtered_gradient = torch.autograd.grad(value, (x, filtered))
        # The original LinearFilter::apply_jacobian applies W, while autograd applies W.T.
        original_rule = weights @ filtered_gradient
        native = x.new_tensor(ref["gradient"])
        diagnostics["original_filter_rule_relative_error"] = (
            (original_rule - native).norm() / native.norm()
        ).item()
    native_gradient = x.new_tensor(ref["gradient"])
    error = ((gradient - native_gradient).norm() / native_gradient.norm()).item()
    direction = gradient.detach() / gradient.norm()
    with torch.no_grad():
        trial = loss(x.new_tensor(ref["trial_x"])).item()
        h = 1e-5
        fd = ((loss(x + h * direction) - loss(x - h * direction)) / (2 * h)).item()
    derivative = gradient.norm().item()
    if case == "material-opt":
        diagnostics["api_gradient"] = gradient.tolist()
        diagnostics["native_gradient"] = native_gradient.tolist()
        with torch.no_grad():
            diagnostics["directional_fd_by_step"] = {
                str(step): (
                    (loss(x + step * direction) - loss(x - step * direction))
                    / (2 * step)
                ).item()
                for step in (1e-4, 1e-3)
            }
    fd_error = abs(fd - derivative) / max(abs(fd), abs(derivative), 1e-30)

    checks = {
        "value": math.isclose(value.item(), ref["value"], rel_tol=1e-8, abs_tol=1e-12),
        "gradient": error < 1e-7,
        "same_trial": math.isclose(
            trial, ref["trial_value"], rel_tol=1e-8, abs_tol=1e-12
        ),
        "directional_fd": fd_error < 1e-3,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parameter_count": x.numel(),
        **diagnostics,
        "value": value.item(),
        "native_value": ref["value"],
        "relative_gradient_l2_error": error,
        "api_trial": trial,
        "native_trial": ref["trial_value"],
        "fd_step": h,
        "directional_ad": derivative,
        "directional_fd": fd,
        "relative_fd_error": fd_error,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=["material-opt", "topology-opt"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = validate(args.case, output)
    except Exception as exc:
        report = {"passed": False, "error": repr(exc)}
    report.update(
        case=args.case,
        source=f"optimizations/{args.case}/run.json",
        commits={
            s: subprocess.check_output(
                ["git", "-C", str(ROOT / s), "rev-parse", "HEAD"], text=True
            ).strip()
            for s in ("polyfem", "differentiability-data")
        },
        scope="Original models; one thread, SimplicialLDLT, redirected output, non-strict native config. No optimizer convergence comparison.",
    )
    write_json(output / "result.json", report)
    print(report, flush=True)
    raise SystemExit(0 if report["passed"] else 1)
