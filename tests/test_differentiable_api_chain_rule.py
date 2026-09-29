from __future__ import annotations

import math
import subprocess
import sys

import pytest


class _QuadraticShapeSession:
    grad_outputs: list[float] = []

    def __init__(self):
        self.vertices = None

    def set_settings(self, settings):
        self.settings = settings

    def set_shape_vertices(self, vertices, *, selection=None):
        self.vertices = vertices
        self.selection = selection

    def set_objective(self, objective):
        self.objective = objective

    def solve(self):
        return self.vertices

    def solve_objective(self):
        return float((self.vertices * self.vertices).sum())

    def backward_shape(self, grad_output):
        grad_output = float(grad_output)
        self.grad_outputs.append(grad_output)
        return 2.0 * self.vertices * grad_output


class _QuadraticShapeBackend:
    DifferentiableSession = _QuadraticShapeSession


def _torch_or_skip():
    completed = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip("torch is not importable in this Python environment")
    return pytest.importorskip("torch")


def _build_vertices(torch, base_vertices, scale):
    return torch.stack(
        (
            base_vertices[:, 0] * scale,
            base_vertices[:, 1],
            base_vertices[:, 2],
        ),
        dim=1,
    )


def _shape_loss(diff, torch, base_vertices, scale):
    diff_model = diff.model([{"geometry": [{"mesh": "unit-test.msh"}]}])
    vertices = _build_vertices(torch, base_vertices, scale)
    shape_opt = diff.shape_opt(
        diff_model,
        vertices,
        objective=diff.Objective.STRESS_NORM,
        backend=_QuadraticShapeBackend,
    )
    return shape_opt()


def test_shapeopt_gradient_chains_to_high_level_torch_parameter():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticShapeSession.grad_outputs = []
    base_vertices = torch.tensor(
        [
            [-1.0, 0.25, 0.0],
            [0.5, -0.25, 1.0],
            [2.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )
    scale = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    loss = _shape_loss(diff, torch, base_vertices, scale)
    loss.backward()

    expected_grad = 2.0 * float(scale.detach()) * float(
        base_vertices[:, 0].square().sum()
    )
    eps = 1e-6
    loss_plus = _shape_loss(
        diff,
        torch,
        base_vertices,
        torch.tensor(float(scale.detach()) + eps, dtype=torch.float64),
    )
    loss_minus = _shape_loss(
        diff,
        torch,
        base_vertices,
        torch.tensor(float(scale.detach()) - eps, dtype=torch.float64),
    )
    finite_difference = float((loss_plus - loss_minus) / (2.0 * eps))

    assert scale.grad is not None
    assert float(scale.grad) == pytest.approx(expected_grad)
    assert float(scale.grad) == pytest.approx(finite_difference, rel=1e-9)
    assert _QuadraticShapeSession.grad_outputs == [1.0]


def test_shapeopt_objective_composition_uses_upstream_grad_output():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticShapeSession.grad_outputs = []
    base_vertices = torch.tensor(
        [
            [-1.0, 0.25, 0.0],
            [0.5, -0.25, 1.0],
            [2.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )
    scale = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    base_loss = _shape_loss(diff, torch, base_vertices, scale)
    composed_loss = base_loss**2
    composed_loss.backward()

    base_grad = 2.0 * float(scale.detach()) * float(
        base_vertices[:, 0].square().sum()
    )
    expected_grad = 2.0 * float(base_loss.detach()) * base_grad

    assert scale.grad is not None
    assert math.isfinite(float(scale.grad))
    assert float(scale.grad) == pytest.approx(expected_grad)
    assert _QuadraticShapeSession.grad_outputs == [
        pytest.approx(2.0 * float(base_loss.detach()))
    ]
