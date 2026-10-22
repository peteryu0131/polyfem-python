"""Small native shape-gradient checks, including cross-body transient contact."""
import meshio
import numpy as np
import pytest
import torch

from polyfempy import differentiable_api as diff
from test_differentiable_api_von_mises import _model
from test_differentiable_api_contact_material import _configuration


@pytest.mark.parametrize("dimension", [2, 3])
def test_static_peak_shape_direction(tmp_path, dimension):
    model, bodies = _model(tmp_path, dimension, "NeoHookean", True)
    vertices = np.concatenate([meshio.read(tmp_path / f"body{i}.msh").points[:, :dimension] for i in range(2)])
    base = torch.tensor(vertices, dtype=torch.float64)
    # Only the interior vertex moves, keeping prescribed boundaries unchanged.
    direction = torch.zeros_like(base)
    direction[4, :2] = torch.tensor([.7, -.3])
    parameter = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    problem = diff.shape_opt(model, parameters={"shift": parameter},
        geometry=lambda shift: base + shift * direction,
        objective=diff.objectives.von_mises_peak(selection=bodies[0], weight=1.7))
    value = problem()
    value.backward()
    analytic = float(parameter.grad)
    h = 1e-5
    with torch.no_grad():
        parameter.add_(h)
        plus = float(problem())
        parameter.sub_(2*h)
        minus = float(problem())
        parameter.add_(h)
    fd = (plus-minus)/(2*h)
    assert abs(fd) > 1e-7
    assert analytic == pytest.approx(fd, rel=2e-3, abs=1e-6)


def test_contact_lower_shape_changes_upper_peak(tmp_path):
    config, lower, upper = _configuration(tmp_path)
    base = torch.tensor(np.concatenate([meshio.read(tmp_path / name).points[:, :2]
                                      for name in ("lower.msh", "upper.msh")]), dtype=torch.float64)
    direction = torch.zeros_like(base)
    # Widen only the lower body, preserving its height and the contact gap.
    direction[:9, 0] = base[:9, 0] - .5
    scale = torch.tensor(.02, dtype=torch.float64, requires_grad=True)
    problem = diff.shape_opt(diff.model(config), parameters={"scale": scale},
        geometry=lambda scale: base + scale * direction,
        objective=diff.objectives.von_mises_peak(selection=upper))
    value = problem()
    value.backward()
    analytic = float(scale.grad)
    h = 1e-4
    with torch.no_grad():
        scale.add_(h)
        plus = float(problem())
        scale.sub_(2*h)
        minus = float(problem())
        scale.add_(h)
    fd = (plus-minus)/(2*h)
    assert abs(fd) > 1e-8
    assert analytic == pytest.approx(fd, rel=2e-2, abs=1e-7)
    torch.testing.assert_close(problem.vertices[9:], base[9:], rtol=0, atol=0)
    print({"loss": float(value.detach()), "adjoint": analytic, "finite_difference": fd})


def test_adaptive_shape_is_rejected_before_forward(tmp_path):
    from polyfempy import polyfempy as backend
    config, _, upper = _configuration(tmp_path)
    config.solver.contact.barrier_stiffness = "adaptive"
    session = backend.DifferentiableSession()
    session.set_settings(config.as_dict())
    with pytest.raises(RuntimeError, match="adaptive contact gradients are not validated"):
        session.set_objective(diff.objectives.von_mises_peak(selection=upper).as_dict())
