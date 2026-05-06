from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from polyfempy.differentiable.design import ParameterizedVertexDesign  # noqa: E402
from polyfempy.differentiable.shape_problem import (  # noqa: E402
    ParameterizedShapeOptimizationProblem,
)


class _FakeResult:
    def __init__(self, vertices):
        self.vertices = vertices
        self.shape_gradient = None

    def release_solver(self):
        pass


def test_parameterized_shape_step_uses_design_names_in_snapshots():
    h = torch.nn.Parameter(torch.tensor(0.04, dtype=torch.float64))
    theta = torch.nn.Parameter(torch.tensor(90.0, dtype=torch.float64))
    base_vertices = torch.zeros((3, 2), dtype=torch.float64)

    def vertex_map(params, base):
        h_value, theta_value = params
        out = base.clone()
        out[:, 0] = out[:, 0] + h_value
        out[:, 1] = out[:, 1] + theta_value * 0.0
        return out

    design = ParameterizedVertexDesign(
        parameters=[h, theta],
        vertex_map=vertex_map,
        base_vertices=base_vertices,
        differentiable_params=["h", "theta_deg"],
    )
    problem = ParameterizedShapeOptimizationProblem(
        design=design,
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        cfg={},
    )
    problem.solve = lambda: _FakeResult(design.vertices())

    optimizer = torch.optim.SGD(design.torch_parameters(), lr=0.1)
    step = next(
        problem.optimize(
            steps=1,
            optimizer=optimizer,
            loss_fn=lambda result: result.vertices.sum(),
        )
    )

    assert set(step.parameter_values_before) == {"h", "theta_deg"}
    assert set(step.parameter_values_after) == {"h", "theta_deg"}
