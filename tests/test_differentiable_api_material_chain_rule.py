from __future__ import annotations

import math
import subprocess
import sys

import pytest


class _QuadraticMaterialSession:
    grad_outputs: list[float] = []
    lame_inputs: list[list[float]] = []

    def __init__(self):
        self.lame = None

    def set_settings(self, settings):
        self.settings = settings

    def set_objective(self, objective):
        self.objective = objective

    def set_material_lame_parameters(self, lame):
        self.lame = lame
        self.lame_inputs.append([float(value) for value in lame])

    def solve_material_objective(self):
        return float((self.lame * self.lame).sum())

    def backward_material(self, grad_output):
        grad_output = float(grad_output)
        self.grad_outputs.append(grad_output)
        return 2.0 * self.lame * grad_output


class _QuadraticMaterialBackend:
    DifferentiableSession = _QuadraticMaterialSession


def _torch_or_skip():
    completed = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip("torch is not importable in this Python environment")
    return pytest.importorskip("torch")


def _material_loss(diff, torch, base_lame, scale):
    diff_model = diff.model([{"geometry": [{"mesh": "unit-test.msh"}]}])
    lame = torch.stack(
        (
            base_lame[:, 0] * scale,
            base_lame[:, 1],
        ),
        dim=1,
    )
    material_opt = diff.material_opt(
        diff_model,
        lame,
        objective=diff.Objective.COMPLIANCE,
        objective_params={"selection": 1},
        backend=_QuadraticMaterialBackend,
    )
    return material_opt()


def test_materialopt_uses_lambda_mu_backend_layout_and_returns_matrix_gradient():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticMaterialSession.grad_outputs = []
    _QuadraticMaterialSession.lame_inputs = []
    diff_model = diff.model([{"geometry": [{"mesh": "unit-test.msh"}]}])
    lame = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = diff.material_opt(
        diff_model,
        lame,
        objective=diff.Objective.COMPLIANCE,
        objective_params={"selection": 1},
        backend=_QuadraticMaterialBackend,
    )()
    loss.backward()

    assert _QuadraticMaterialSession.lame_inputs == [
        [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    ]
    assert lame.grad is not None
    assert tuple(lame.grad.shape) == (3, 2)
    expected_grad = torch.tensor(
        [
            [2.0, 20.0],
            [4.0, 40.0],
            [6.0, 60.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(lame.grad, expected_grad)


def test_materialopt_gradient_chains_to_high_level_torch_parameter():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticMaterialSession.grad_outputs = []
    _QuadraticMaterialSession.lame_inputs = []
    base_lame = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
    )
    scale = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    loss = _material_loss(diff, torch, base_lame, scale)
    loss.backward()

    expected_grad = 2.0 * float(scale.detach()) * float(
        base_lame[:, 0].square().sum()
    )
    eps = 1e-6
    loss_plus = _material_loss(
        diff,
        torch,
        base_lame,
        torch.tensor(float(scale.detach()) + eps, dtype=torch.float64),
    )
    loss_minus = _material_loss(
        diff,
        torch,
        base_lame,
        torch.tensor(float(scale.detach()) - eps, dtype=torch.float64),
    )
    finite_difference = float((loss_plus - loss_minus) / (2.0 * eps))

    assert scale.grad is not None
    assert float(scale.grad) == pytest.approx(expected_grad)
    assert float(scale.grad) == pytest.approx(finite_difference, rel=1e-9)
    assert _QuadraticMaterialSession.grad_outputs[0] == 1.0


def test_materialopt_objective_composition_uses_upstream_grad_output():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticMaterialSession.grad_outputs = []
    base_lame = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
    )
    scale = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    base_loss = _material_loss(diff, torch, base_lame, scale)
    composed_loss = base_loss**2
    composed_loss.backward()

    base_grad = 2.0 * float(scale.detach()) * float(
        base_lame[:, 0].square().sum()
    )
    expected_grad = 2.0 * float(base_loss.detach()) * base_grad

    assert scale.grad is not None
    assert math.isfinite(float(scale.grad))
    assert float(scale.grad) == pytest.approx(expected_grad)
    assert _QuadraticMaterialSession.grad_outputs == [
        pytest.approx(2.0 * float(base_loss.detach()))
    ]
