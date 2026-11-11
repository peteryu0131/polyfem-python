"""Check the material-opt directional derivative using original OptState evaluations."""

import argparse
import json
from pathlib import Path

import torch

from .compare_original_json import run_native, write_json
from .validate_optimization_runs import stage


def check(output):
    output.mkdir(parents=True, exist_ok=False)
    spec, _ = stage("material-opt", output)
    path = output / "opt.json"

    def evaluate(design, name):
        spec["parameters"][0]["initial"] = design.tolist()
        write_json(path, spec)
        result = output / f"{name}.json"
        run_native(
            "polyfem_reference_probe.exe",
            [path, result, "--no-strict-validation"],
            output / f"{name}.log",
        )
        return json.loads(result.read_text())

    x = torch.tensor(spec["parameters"][0]["initial"], dtype=torch.float64)
    base = evaluate(x, "baseline")
    gradient = x.new_tensor(base["gradient"])
    direction = gradient / gradient.norm()
    h = 1e-4
    plus = evaluate(x + h * direction, "plus")["value"]
    minus = evaluate(x - h * direction, "minus")["value"]
    fd = (plus - minus) / (2 * h)
    report = dict(
        source="optimizations/material-opt/run.json",
        x=x.tolist(),
        direction=direction.tolist(),
        step=h,
        plus=plus,
        minus=minus,
        directional_fd=fd,
        directional_ad=gradient.norm().item(),
        relative_error=abs(fd - gradient.norm().item())
        / max(abs(fd), gradient.norm().item()),
    )
    write_json(output / "result.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    check(parser.parse_args().output.resolve())
