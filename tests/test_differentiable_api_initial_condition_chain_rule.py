from __future__ import annotations

import math
import subprocess
import sys

import pytest


class _QuadraticInitialConditionSession:
    grad_outputs: list[float] = []
    initial_condition_inputs: list[list[float]] = []

    def __init__(self):
        self.initial_condition = None

    def set_settings(self, settings):
        self.settings = settings

    def set_objective(self, objective):
        self.objective = objective

    def initial_condition_dof_count(self):
        return 3

    def set_initial_condition_parameters(self, initial_condition):
        self.initial_condition = initial_condition
        self.initial_condition_inputs.append(
            [float(value) for value in initial_condition]
        )

    def solve_initial_condition_objective(self):
        return float((self.initial_condition * self.initial_condition).sum())

    def backward_initial_condition(self, grad_output):
        grad_output = float(grad_output)
        self.grad_outputs.append(grad_output)
        return 2.0 * self.initial_condition * grad_output


class _QuadraticInitialConditionBackend:
    DifferentiableSession = _QuadraticInitialConditionSession


def _transient_objective():
    return {
        "type": "transient_integral",
        "state": 0,
        "integral_type": "final",
        "steps": [],
        "weight": 1.0,
        "print_energy": "",
        "static_objective": {
            "type": "stress_norm",
            "state": 0,
            "volume_selection": [],
            "power": 2,
            "weight": 1.0,
            "print_energy": "",
        },
    }


def _torch_or_skip():
    completed = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip("torch is not importable in this Python environment")
    return pytest.importorskip("torch")


def _initial_condition_loss(diff, torch, base_u0, base_v0, speed):
    diff_model = diff.model([{"geometry": [{"mesh": "unit-test.msh"}]}])
    initial_condition = torch.stack(
        (
            base_u0,
            base_v0 * speed,
        ),
        dim=1,
    )
    initial_opt = diff.initial_condition_opt(
        diff_model,
        initial_condition,
        objective=_transient_objective(),
        backend=_QuadraticInitialConditionBackend,
    )
    return initial_opt()


def test_initialconditionopt_uses_solution_velocity_backend_layout_and_returns_matrix_gradient():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticInitialConditionSession.grad_outputs = []
    _QuadraticInitialConditionSession.initial_condition_inputs = []
    diff_model = diff.model([{"geometry": [{"mesh": "unit-test.msh"}]}])
    initial_condition = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = diff.initial_condition_opt(
        diff_model,
        initial_condition,
        objective=diff.Objective.STRESS_NORM,
        objective_params={"power": 2},
        backend=_QuadraticInitialConditionBackend,
    )()
    loss.backward()

    assert _QuadraticInitialConditionSession.initial_condition_inputs == [
        [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    ]
    assert initial_condition.grad is not None
    assert tuple(initial_condition.grad.shape) == (3, 2)
    expected_grad = torch.tensor(
        [
            [2.0, 20.0],
            [4.0, 40.0],
            [6.0, 60.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(initial_condition.grad, expected_grad)


def test_initialconditionopt_gradient_chains_to_high_level_velocity_parameter():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticInitialConditionSession.grad_outputs = []
    _QuadraticInitialConditionSession.initial_condition_inputs = []
    base_u0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    base_v0 = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    speed = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    loss = _initial_condition_loss(diff, torch, base_u0, base_v0, speed)
    loss.backward()

    expected_grad = 2.0 * float(speed.detach()) * float(base_v0.square().sum())
    eps = 1e-6
    loss_plus = _initial_condition_loss(
        diff,
        torch,
        base_u0,
        base_v0,
        torch.tensor(float(speed.detach()) + eps, dtype=torch.float64),
    )
    loss_minus = _initial_condition_loss(
        diff,
        torch,
        base_u0,
        base_v0,
        torch.tensor(float(speed.detach()) - eps, dtype=torch.float64),
    )
    finite_difference = float((loss_plus - loss_minus) / (2.0 * eps))

    assert speed.grad is not None
    assert float(speed.grad) == pytest.approx(expected_grad)
    assert float(speed.grad) == pytest.approx(finite_difference, rel=1e-9)
    assert _QuadraticInitialConditionSession.grad_outputs[0] == 1.0


def test_initialconditionopt_objective_composition_uses_upstream_grad_output():
    torch = _torch_or_skip()
    from polyfempy import differentiable_api as diff

    _QuadraticInitialConditionSession.grad_outputs = []
    base_u0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    base_v0 = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    speed = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    base_loss = _initial_condition_loss(diff, torch, base_u0, base_v0, speed)
    composed_loss = base_loss**2
    composed_loss.backward()

    base_grad = 2.0 * float(speed.detach()) * float(base_v0.square().sum())
    expected_grad = 2.0 * float(base_loss.detach()) * base_grad

    assert speed.grad is not None
    assert math.isfinite(float(speed.grad))
    assert float(speed.grad) == pytest.approx(expected_grad)
    assert _QuadraticInitialConditionSession.grad_outputs == [
        pytest.approx(2.0 * float(base_loss.detach()))
    ]
