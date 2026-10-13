"""Real-backend checks for the constant-gravity, equal-height benchmark."""

import ast
import json
import math
from pathlib import Path

import pytest


def _load_example(**settings):
    """Run the actual script with edited top-level constants, as a user would.

    Only test code does this substitution: no alternative forward model or
    optimization loop is used to test changes to the example's settings.
    """
    pytest.importorskip("torch")
    pytest.importorskip("polyfempy.polyfempy")
    path = Path(__file__).resolve().parents[1] / "differentiable_example/initial_condition/projectile_angle.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in settings:
                node.value = ast.Constant(settings[target.id])
                changed.add(target.id)
    assert changed == set(settings), "Test settings must refer to real example constants"
    ast.fix_missing_locations(tree)
    namespace = {"__name__": "projectile_test", "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def example():
    return _load_example()


def _position_at(example, theta, *, end_time=None):
    """Independent probe used by the physics tests, not by the user example."""
    import torch
    from polyfempy import differentiable_api as diff

    payload = example["diff_model"].as_dicts()[0]
    if end_time is not None:
        payload["time"]["tend"] = end_time
    model = diff.model(payload)
    velocity = example["SPEED"] * torch.stack((torch.cos(theta), torch.sin(theta)))
    v0 = velocity.repeat(example["dof_count"] // 2)
    initial = torch.stack((torch.zeros_like(v0), v0), dim=1)
    return torch.stack([
        diff.initial_condition_opt(
            model, initial, objective=diff.objectives.position(dim=dim).final(),
        )()
        for dim in range(2)
    ])


def test_projectile_range_and_gradient_match_discrete_motion_and_fd(example):
    import torch

    theta = torch.tensor(math.pi / 6, dtype=torch.float64, requires_grad=True)
    position = _position_at(example, theta.detach())
    t, dt = example["PROBE_TIME"], example["DT"]
    # Independent implicit-Euler solution under unit gravity and unit speed.
    expected_x = math.cos(math.pi / 6) * t
    expected_y = math.sin(math.pi / 6) * t - t * (t + dt) / 2
    assert position.detach().tolist() == pytest.approx([expected_x, expected_y], abs=1e-7)
    value = example["loss"](theta)
    expected_range = math.sin(math.pi / 3) - dt * math.cos(math.pi / 6)
    assert -float(value.detach()) == pytest.approx(expected_range, abs=1e-6)
    value.backward()
    # Includes the derivative of flight time, not only horizontal speed.
    expected_grad = -2 * math.cos(math.pi / 3) - dt * math.sin(math.pi / 6)
    assert theta.grad.shape == torch.Size([])
    assert float(theta.grad) == pytest.approx(expected_grad, rel=1e-5, abs=1e-7)
    eps = 1e-4
    upper = example["loss"](theta.detach() + eps)
    lower = example["loss"](theta.detach() - eps)
    fd = float((upper - lower).detach()) / (2 * eps)
    assert float(theta.grad) == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_projectile_range_converges_to_continuous_reference(example, monkeypatch):
    import torch
    from polyfempy import differentiable_api as diff

    theta = torch.tensor(math.pi / 4, dtype=torch.float64)
    errors = []
    for dt in (0.04, 0.02):
        payload = example["diff_model"].as_dicts()[0]
        payload["time"]["dt"] = dt
        monkeypatch.setitem(example, "diff_model", diff.model(payload))
        errors.append(abs(-float(example["loss"](theta)) - 1.0))
    assert 0 < errors[1] < errors[0]
    assert errors[1] / errors[0] == pytest.approx(0.5, rel=1e-3)


def test_reconstructed_landing_agrees_with_longer_backend_trajectory(example):
    import torch

    theta = torch.tensor(math.pi / 6, dtype=torch.float64)
    # At 30 degrees the BDF1 return time is 0.99 s. Independently simulate
    # just before/after it, well beyond the 0.2 s reconstruction probe.
    positions = []
    for end_time in (0.98, 1.0):
        positions.append(_position_at(example, theta, end_time=end_time))
    before, after = positions
    assert float(before[1]) > 0 > float(after[1])
    fraction = before[1] / (before[1] - after[1])
    crossing_x = before[0] + fraction * (after[0] - before[0])
    expected_range = math.cos(math.pi / 6) * 0.99
    # Linear interpolation of the sampled parabola has O(dt^2) error.
    assert float(crossing_x) == pytest.approx(expected_range, abs=2e-4)


@pytest.mark.parametrize(
    "angle, speed, gravity",
    [(20.0, 2.0, 1.0), (5.0, 0.5, 2.0), (85.0, 1.0, 0.5), (70.0, 1.0, 9.81)],
    ids=["faster", "slower-stronger-gravity", "high-angle-low-gravity", "earth-gravity"],
)
def test_changed_settings_preserve_gradient_and_optimization(angle, speed, gravity, tmp_path, capsys):
    import torch

    case = _load_example(START_ANGLE_DEGREES=angle, SPEED=speed, GRAVITY=gravity)
    case["OUTPUT_DIR"] = tmp_path
    theta = torch.tensor(math.radians(angle), dtype=torch.float64)
    eps = 1e-4
    upper = case["loss"](theta + eps)
    lower = case["loss"](theta - eps)
    finite_difference = float(upper - lower) / (2 * eps)

    # Exercise the actual user-facing loop, including its step scaling and save.
    case["main"]()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    history = summary["history"]
    dt = case["DT"]
    scale = speed ** 2 / gravity
    expected_gradient = -2 * scale * math.cos(2 * float(theta)) - speed * dt * math.sin(float(theta))
    assert history[0]["angle_gradient"] == pytest.approx(expected_gradient, rel=1e-5, abs=1e-7)
    assert history[0]["angle_gradient"] == pytest.approx(finite_difference, rel=1e-4, abs=1e-6)

    # Independent BDF1 solution: R = s^2/g*sin(2*theta) - s*dt*cos(theta).
    for row in history + [{"angle_degrees": summary["angle_degrees"], "range": summary["range"]}]:
        radians = math.radians(row["angle_degrees"])
        expected_range = scale * math.sin(2 * radians) - speed * dt * math.cos(radians)
        assert row["range"] == pytest.approx(expected_range, rel=1e-6, abs=1e-7)
    ranges = [row["range"] for row in history] + [summary["range"]]
    assert all(after >= before - 1e-8 * scale for before, after in zip(ranges, ranges[1:]))
    assert ranges[-1] > ranges[0]

    # Solve dR/dtheta=0 independently; finite dt shifts the optimum above 45 deg.
    ratio = gravity * dt / speed
    sin_optimum = (ratio + math.sqrt(ratio ** 2 + 32)) / 8
    expected_angle = math.degrees(math.asin(sin_optimum))
    assert summary["angle_degrees"] == pytest.approx(expected_angle, abs=0.02)
    assert len(history) == case["STEPS"]
    assert summary["speed"] == speed
    assert summary["gravity"] == gravity
    assert summary["start_angle_degrees"] == angle
    assert summary["effective_learning_rate"] == pytest.approx(case["LEARNING_RATE"] / scale)
    assert "Results saved to" in capsys.readouterr().out


@pytest.mark.parametrize("settings, message", [
    ({"SPEED": 0.0}, "SPEED must be finite and positive"),
    ({"SPEED": float("nan")}, "SPEED must be finite and positive"),
    ({"GRAVITY": -1.0}, "GRAVITY must be finite and positive"),
    ({"GRAVITY": float("inf")}, "GRAVITY must be finite and positive"),
    ({"DT": 0.0}, "DT must be finite and positive"),
    ({"PROBE_TIME": 0.0}, "PROBE_TIME must be finite and positive"),
    ({"PROBE_TIME": 0.205}, "integer multiple of DT"),
    ({"PROBE_TIME": 0.005}, "integer multiple of DT"),
    ({"START_ANGLE_DEGREES": 0.0}, "between 5 and 85"),
    ({"START_ANGLE_DEGREES": 90.0}, "between 5 and 85"),
    ({"START_ANGLE_DEGREES": float("nan")}, "between 5 and 85"),
    ({"STEPS": 0}, "positive integer"),
    ({"STEPS": 1.5}, "positive integer"),
    ({"LEARNING_RATE": 0.0}, "LEARNING_RATE must be finite and positive"),
    ({"SPEED": 0.1, "GRAVITY": 9.81}, "DT is too large"),
])
def test_invalid_settings_fail_before_backend_preparation(settings, message, monkeypatch):
    from polyfempy import differentiable_api as diff

    def unexpected_preparation(self):
        pytest.fail("Invalid settings must be rejected before backend preparation")

    monkeypatch.setattr(diff.DifferentiableModel, "metadata", unexpected_preparation)
    with pytest.raises(ValueError, match=message):
        _load_example(**settings)
