"""Match the selected surface on body 2 while moving body 1."""

import torch

from polyfempy import polyfempy as backend


def bernstein(points):
    t = (points - points.min()) / (points.max() - points.min())
    return torch.stack(
        ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3), dim=1
    )


def mesh_and_boundary_nodes(config, boundary_id):
    session = backend.DifferentiableSession()
    session.set_settings(config)
    vertices = torch.tensor(session.mesh_vertices(), dtype=torch.float64)
    nodes = torch.tensor(session.boundary_vertex_ids([boundary_id], False), dtype=torch.long)
    return vertices, nodes


def trajectory_solver_settings(states):
    for state in states:
        state["solver"]["nonlinear"].update(rel_grad_norm_tol=1e-7, grad_norm_tol=1e-5)
        state["solver"]["augmented_lagrangian"] = {
            "nonlinear": {"rel_grad_norm_tol": 1e-7, "grad_norm_tol": 1e-5}
        }
