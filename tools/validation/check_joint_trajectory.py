"""Check both input blocks of the joint shape and displacement example."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import polyfem.diff as diff

from .optimization_reference import load
from differentiable_example.optimizations.shape_matching.multiparameter_sdf_trajectory import build_problem

from .compare_original_json import write_json


def problem(spec, states):
    (config,) = states
    trajectory, smooth, barrier = spec["functionals"]
    target = diff.objectives.sdf_target(
        surface_selection=trajectory["surface_selection"],
        control_points=trajectory["control_points"],
        knots=trajectory["knots"],
        delta=trajectory["delta"][0],
    )
    return build_problem(
        config, target, barrier["dhat"], barrier["weight"], smooth["weight"]
    )


def check(output, time_steps=()):
    case = "multiparameter-sdf-trajectory-surface-opt"
    with TemporaryDirectory() as tmp:
        initial, loss = problem(*load(case, Path(tmp)))
        x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        value = loss(x)
        (gradient,) = torch.autograd.grad(value, x)
        checks = []
        directions = []
        for label, section in (("shape", slice(0, 4)), ("motion", slice(4, None))):
            direction = torch.zeros_like(x)
            direction[section] = gradient[section] / gradient[section].norm()
            directions.append((label, direction))
        for step in time_steps:
            if not 1 <= step <= len(x) - 4:
                raise ValueError(f"Expected a motion step between 1 and {len(x) - 4}")
            direction = torch.zeros_like(x)
            direction[4 + step - 1] = 1
            directions.append((f"motion_step_{step}", direction))
        for label, direction in directions:
            ad = gradient.dot(direction).item()
            for h in (1e-4, 3e-5):
                with torch.no_grad():
                    fd = (
                        (loss(x + h * direction) - loss(x - h * direction)) / (2 * h)
                    ).item()
                checks.append(
                    dict(
                        group=label,
                        step=h,
                        ad=ad,
                        fd=fd,
                        relative_error=abs(fd - ad) / max(abs(ad), abs(fd), 1e-30),
                    )
                )
        write_json(
            output,
            dict(
                case=case,
                value=value.item(),
                checks=checks,
                rtol=1e-3,
                passed=all(c["relative_error"] < 1e-3 for c in checks),
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--time-steps",
        type=int,
        nargs="+",
        default=(),
        help="Individual motion steps to check, numbered from 1",
    )
    args = parser.parse_args()
    check(args.output, args.time_steps)
