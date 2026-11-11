"""Compare gradients of the original surface-matching runs to central differences."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from differentiable_example.optimizations.material_and_contact import damping_opt, friction_opt
from .optimization_reference import load, material_loss, target_objective

from .compare_original_json import write_json

CASES = {
    "damping-opt": damping_opt,
    "friction-opt": friction_opt,
}


def check(case, output, bdf1=False, frictionless=False):
    with TemporaryDirectory() as root:
        spec, states = load(case, Path(root))
        if frictionless:
            if case == "friction-opt":
                raise ValueError("Friction is the design variable in friction-opt")
            for state in states:
                state["contact"]["friction_coefficient"] = 0.0
        if bdf1:
            for state in states:
                state["time"]["integrator"]["steps"] = 1
        if case == "material-opt":
            initial = spec["parameters"][0]["initial"]
            loss = material_loss(spec, states)
        else:
            initial, loss = CASES[case].build_problem(*states, target_objective(spec))
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        value = loss(x)
        (gradient,) = torch.autograd.grad(value, x)
        direction = gradient / gradient.norm()
        results = []
        with torch.no_grad():
            for h in (1e-3, 1e-4):
                fd = (
                    (loss(x + h * direction) - loss(x - h * direction)) / (2 * h)
                ).item()
                ad = gradient.norm().item()
                results.append(
                    dict(
                        step=h,
                        fd=fd,
                        ad=ad,
                        relative_error=abs(fd - ad) / max(abs(fd), abs(ad), 1e-30),
                    )
                )
        write_json(
            output,
            dict(
                case=case,
                friction_coefficients=[
                    s["contact"]["friction_coefficient"] for s in states
                ],
                bdf_steps=states[0]["time"]["integrator"]["steps"],
                value=value.item(),
                gradient=gradient.tolist(),
                direction=direction.tolist(),
                checks=results,
                rtol=1e-3,
                passed=all(r["relative_error"] < 1e-3 for r in results),
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[*CASES, "material-opt"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--bdf1", action="store_true", help="Use one-step BDF in both forward models"
    )
    parser.add_argument(
        "--frictionless",
        action="store_true",
        help="Set friction to zero in both material or damping simulations",
    )
    args = parser.parse_args()
    check(args.case, args.output, args.bdf1, args.frictionless)
