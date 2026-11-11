"""Compare the original PolyFEM optimization entry with the callable API.

The source JSON and submodules are read-only. Staged copies change only paths,
execution/output settings, and compute_objective (no optimizer iterations).
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
import polyfem.diff as diff


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "build/temp.win-amd64-cpython-311/Release/tools/optimization_reference/Release"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def stage(kind, directory):
    name = "topology-compliance" if kind == "material" else "neohookean-stress-3d"
    source = ROOT / "differentiability-data/input" / f"{name}-opt.json"
    config = json.loads(source.read_text())
    forward_source = source.parent / config["states"][0]["path"]
    forward = json.loads(forward_source.read_text())
    for entry in forward["geometry"]:
        entry["mesh"] = str((forward_source.parent / entry["mesh"]).resolve())
    forward["solver"]["max_threads"] = 1
    forward["solver"]["linear"]["solver"] = "Eigen::SimplicialLDLT"
    forward["output"] = {
        "directory": str(directory), "json": "",
        "paraview": {"file_name": ""},
        "advanced": {"save_time_sequence": False},
    }
    write_json(directory / "state.json", forward)
    config["states"][0]["path"] = str(directory / "state.json")
    config["compute_objective"] = True
    config["solver"]["max_threads"] = 1
    config["output"] = {"directory": str(directory),
                        "log": {"level": "info", "path": "", "quiet": False}}
    write_json(directory / "opt.json", config)
    return source


def run_native(executable, arguments, log):
    env = os.environ.copy()
    env["PATH"] = str(ROOT / "polyfempy") + os.pathsep + env["PATH"]
    command = [str(TOOLS / executable), *map(str, arguments)]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=log.parent, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{executable} failed ({result.returncode}); see {log}")
    return command


def compare(kind, directory, *, strict=True):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = stage(kind, directory)
    cli_args = ["--json", directory / "opt.json"]
    probe_args = [directory / "opt.json", directory / "reference.json"]
    if not strict:
        cli_args.append("--no_strict_validation")
        probe_args.append("--no-strict-validation")
    cli = run_native("polyfem_reference_cli.exe", cli_args, directory / "cli.log")
    probe = run_native("polyfem_reference_probe.exe", probe_args, directory / "probe.log")
    reference = json.loads((directory / "reference.json").read_text())
    cli_values = re.findall(r"Objective is ([\d.eE+\-]+)", (directory / "cli.log").read_text())
    assert cli_values, "Original CLI did not report its objective"
    np.testing.assert_allclose(float(cli_values[-1]), reference["value"], rtol=1e-7, atol=1e-9)
    config = reference["state_args"][0]
    objective = reference["resolved_opt"]["functionals"][0]
    x = torch.tensor(reference["x"], dtype=torch.float64, requires_grad=True)
    if kind == "material":
        obj = diff.material(config, body=1, objective=objective)

        def loss(x):
            return obj(E=x.pow(5), nu=.3)

        E = x.detach().pow(5)
        mapped = torch.cat((E * .3 / (1-.3**2), E / (2*(1+.3))))
    else:
        obj = diff.shape(config, objective=objective)

        def loss(x):
            return obj(V=x.reshape(-1, 3))

        mapped = x.detach()
    np.testing.assert_allclose(mapped.numpy(), reference["mapped_parameters"], rtol=1e-12, atol=1e-9)
    value = loss(x)
    value.backward()
    gradient = x.grad.detach().numpy()
    expected = np.asarray(reference["gradient"])
    with torch.no_grad():
        trial_value = float(loss(x.new_tensor(reference["trial_x"])))
    report = {
        "case": kind, "source": str(source.relative_to(ROOT)),
        "strict_json_validation": strict,
        "cli_value": float(cli_values[-1]),
        "commits": {name: subprocess.check_output(["git", "-C", str(ROOT / name), "rev-parse", "HEAD"], text=True).strip()
                    for name in ("polyfem", "differentiability-data")},
        "commands": [cli, probe], "parameter_count": x.numel(),
        "initial_range": [float(x.detach().min()), float(x.detach().max())],
        "mapping_max_abs_error": float(np.max(np.abs(mapped.numpy() - np.asarray(reference["mapped_parameters"])))),
        "original_value": reference["value"], "api_value": float(value.detach()),
        "gradient_max_abs_error": float(np.max(np.abs(gradient-expected))),
        "gradient_relative_l2_error": float(np.linalg.norm(gradient-expected)/np.linalg.norm(expected)),
        "original_trial_value": reference["trial_value"], "api_trial_value": trial_value,
        "trial_kind": "same normalized-gradient trial point, not original optimizer update",
        "adaptations": ["absolute paths", "one thread", "SimplicialLDLT", "output settings", "compute_objective=true"],
        "gradient": gradient.tolist(), "reference_gradient": expected.tolist(),
    }
    write_json(directory / "comparison.json", report)
    np.testing.assert_allclose(float(value.detach()), reference["value"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(trial_value, reference["trial_value"], rtol=1e-7, atol=1e-9)
    print(json.dumps({key: value for key, value in report.items() if "gradient" not in key or not isinstance(value, list)}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["material", "shape"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--tools", type=Path, default=TOOLS)
    parser.add_argument("--no-strict-validation", action="store_true")
    options = parser.parse_args()
    TOOLS = options.tools.resolve()
    compare(options.case, options.output, strict=not options.no_strict_validation)
