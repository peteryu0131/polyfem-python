"""Shared data paths, solver settings, and output options for the examples."""

import argparse
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import polyfem

DATA_DIR = Path(__file__).resolve().parents[1] / "differentiability-data"
OPTIMIZATIONS = DATA_DIR / "optimizations"


def example_output(directory=None):
    return polyfem.output(
        directory=None if directory is None else str(directory),
        json="",
        log=polyfem.log(level="warning"),
        paraview=polyfem.output_paraview(file_name=""),
        advanced=polyfem.output_advanced(save_time_sequence=False),
    )


def solver_config(config):
    """Keep global materials and scalar transforms in the solver's native input form."""
    scene = config.as_dict()
    materials = scene["materials"]
    if len(materials) == 1 and "id" not in materials[0]:
        scene["materials"] = materials[0]
    for mesh in scene["geometry"]:
        # The reference configurations use the implicit mesh type.
        if mesh.get("type") == "mesh":
            del mesh["type"]
        transform = mesh.get("transformation", {})
        for key in ("rotation", "scale"):
            value = transform.get(key)
            if isinstance(value, list) and len(value) == 1:
                transform[key] = value[0]
    return scene


def control_solver(**kwargs):
    return polyfem.solver(
        max_threads=1,
        linear=polyfem.linear(solver="Eigen::SimplicialLDLT"),
        nonlinear=polyfem.nonlinear(grad_norm_tol=1e-10, rel_grad_norm_tol=0),
        **kwargs,
    )


def bdf1_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bdf1", action="store_true", help="Use the gradient-checked BDF1 variant"
    )
    return parser.parse_args().bdf1


@contextmanager
def simulation_directory(example_file=None):
    """Run in a temporary directory, or keep exports in examples/output/<example>."""
    if example_file is None:
        workspace = TemporaryDirectory()
    else:
        output = DATA_DIR.parent / "examples" / "output" / Path(example_file).stem
        output.mkdir(parents=True, exist_ok=True)
        workspace = nullcontext(output)
    previous = Path.cwd()
    with workspace as directory:
        try:
            os.chdir(directory)
            yield Path(directory)
        finally:
            os.chdir(previous)


def mesh_vertices(config):
    import torch
    from polyfempy import polyfempy as backend

    session = backend.DifferentiableSession()
    session.set_settings(config)
    return torch.tensor(session.mesh_vertices(), dtype=torch.float64)


def initial_state(config):
    import torch
    from polyfempy import polyfempy as backend

    session = backend.DifferentiableSession()
    session.set_settings(config)
    values = torch.tensor(session.initial_condition_parameters(), dtype=torch.float64)
    return values.reshape(2, -1).T.contiguous()
