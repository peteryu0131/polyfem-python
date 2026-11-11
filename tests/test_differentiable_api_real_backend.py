from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _cube_vertices(torch):
    return torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def _laplacian_shape_smoke_settings(mesh_path: Path, output_dir: Path) -> dict:
    return {
        "geometry": {
            "mesh": str(mesh_path),
            "surface_selection": [
                {
                    "id": 1,
                    "box": [
                        [-0.501, -0.501, -0.501],
                        [-0.499, 0.501, 0.501],
                    ],
                    "relative": False,
                },
                {
                    "id": 3,
                    "box": [
                        [0.499, -0.501, -0.501],
                        [0.501, 0.501, 0.501],
                    ],
                    "relative": False,
                },
            ],
            "advanced": {"normalize_mesh": False},
        },
        "materials": {"type": "Laplacian"},
        "preset_problem": {"type": "Franke"},
        "solver": {
            "max_threads": 1,
            "linear": {"solver": "Eigen::SimplicialLDLT"},
            "nonlinear": {"line_search": {"use_grad_norm_tol": 1e-5}},
        },
        "time": {"tend": 1, "dt": 1},
        "output": {
            "directory": str(output_dir),
            "json": "",
            "paraview": {"file_name": ""},
            "advanced": {"save_time_sequence": False},
        },
    }


def _bar_face_count(mesh_path: Path) -> int:
    return sum(
        1
        for line in mesh_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("f ")
    )


def _linear_elasticity_compliance_settings(mesh_path: Path, output_dir: Path) -> dict:
    return {
        "geometry": [
            {
                "mesh": str(mesh_path),
                "surface_selection": {"threshold": 0.0001},
                "volume_selection": 1,
            }
        ],
        "space": {
            "discr_order": 1,
            "advanced": {
                "quadrature_order": 2,
                "mass_quadrature_order": 2,
            },
        },
        "solver": {
            "max_threads": 1,
            "linear": {"solver": "Eigen::SimplicialLDLT"},
            "advanced": {
                "characteristic_force_density": 1,
                "characteristic_length": 1,
            },
            "nonlinear": {
                "norm_type": "Euclidean",
                "rel_grad_norm_tol": 0,
                "first_grad_norm_tol": 1e-10,
                "grad_norm_tol": 1e-8,
            },
        },
        "boundary_conditions": {
            "rhs": [10, 100],
            "dirichlet_boundary": [
                {
                    "id": 1,
                    "value": [0.0, 0.0],
                }
            ],
        },
        "materials": {
            "type": "LinearElasticity",
            "lambda": 1.0e5,
            "mu": 5.0e4,
        },
        "output": {
            "directory": str(output_dir),
            "json": "",
            "paraview": {"file_name": ""},
            "advanced": {"save_time_sequence": False},
        },
    }


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _transient_elastic_initial_condition_settings(
    output_dir: Path,
) -> dict:
    from polyfempy.runtime._solve_contract import prepare_canonical_solve_input

    config_path = (
        ROOT / "polyfem-data" / "contact" / "examples" / "2D"
        / "initial_angular_velocity.json"
    )
    common_path = (config_path.parent / ".." / "common.json").resolve()
    common = json.loads(common_path.read_text(encoding="utf-8"))
    canonical = prepare_canonical_solve_input(
        vertices=None,
        cells=None,
        cfg=config_path,
    )

    payload = _deep_merge_dict(common, canonical.backend_settings)
    payload["geometry"] = canonical.backend_settings["geometry"]
    payload.pop("initial_conditions", None)
    payload["time"]["dt"] = 0.04
    payload["time"]["tend"] = 0.04
    payload["time"]["integrator"] = "BDF"
    payload["solver"]["max_threads"] = 1
    payload["output"]["directory"] = str(output_dir)
    payload["output"]["json"] = ""
    payload["output"]["paraview"]["file_name"] = ""
    payload["output"]["advanced"]["save_time_sequence"] = False
    return payload


def _material_loss(diff, diff_model, lame, backend):
    return diff.material_opt(
        diff_model,
        lame,
        objective=diff.Objective.COMPLIANCE,
        objective_params={"selection": 1},
        backend=backend,
    )()


def _initial_condition_objective() -> dict:
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


def _initial_condition_loss(diff, diff_model, initial_condition, backend):
    return diff.initial_condition_opt(
        diff_model,
        initial_condition,
        objective=_initial_condition_objective(),
        backend=backend,
    )()


def _initial_condition_dof_count(backend, payload: dict) -> int:
    session = backend.DifferentiableSession()
    session.set_settings(payload)
    return int(session.initial_condition_dof_count())


def test_shapeopt_real_backend_forward_backward_smoke(tmp_path):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = ROOT / "polyfem-data" / "contact" / "meshes" / "3D" / "simple" / "cube.msh"
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _laplacian_shape_smoke_settings(mesh_path, tmp_path),
    ])
    x = _cube_vertices(torch)

    u = diff.ShapeOpt.apply(diff_model, "all", x, backend)
    loss = u.square().mean()
    loss.backward()

    assert tuple(u.shape) == (8, 1)
    assert u.dtype is torch.float64
    assert u.device == x.device
    assert torch.isfinite(u).all()
    assert x.grad is not None
    assert tuple(x.grad.shape) == tuple(x.shape)
    assert torch.isfinite(x.grad).all()


def test_shapeopt_real_backend_objective_backward_smoke(tmp_path):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = ROOT / "polyfem-data" / "contact" / "meshes" / "3D" / "simple" / "cube.msh"
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _laplacian_shape_smoke_settings(mesh_path, tmp_path),
    ])
    x = _cube_vertices(torch)

    loss = diff.ShapeOpt.apply(
        model=diff_model,
        selection="all",
        tensor=x,
        objective=diff.Objective.STRESS_NORM,
        objective_params={
            "selection": 1,
            "power": 8,
        },
        backend=backend,
    )
    loss.backward()

    assert tuple(loss.shape) == ()
    assert loss.dtype is torch.float64
    assert loss.device == x.device
    assert torch.isfinite(loss)
    assert x.grad is not None
    assert tuple(x.grad.shape) == tuple(x.shape)
    assert torch.isfinite(x.grad).all()
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def test_materialopt_real_backend_compliance_gradient_and_chain_rule(tmp_path):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    mesh_path = (
        ROOT
        / "polyfem-data"
        / "contact"
        / "meshes"
        / "2D"
        / "simple"
        / "bar"
        / "bar40.obj"
    )
    if not mesh_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    diff_model = diff.model([
        _linear_elasticity_compliance_settings(mesh_path, tmp_path),
    ])
    n_elements = _bar_face_count(mesh_path)
    base_lame = torch.empty((n_elements, 2), dtype=torch.float64)
    base_lame[:, 0] = 1.0e5
    base_lame[:, 1] = 5.0e4

    lame_leaf = base_lame.clone().requires_grad_(True)
    primitive_loss = _material_loss(diff, diff_model, lame_leaf, backend)
    primitive_loss.backward()

    assert tuple(primitive_loss.shape) == ()
    assert primitive_loss.dtype is torch.float64
    assert torch.isfinite(primitive_loss)
    assert lame_leaf.grad is not None
    assert tuple(lame_leaf.grad.shape) == (n_elements, 2)
    assert torch.isfinite(lame_leaf.grad).all()

    direction = torch.zeros_like(base_lame)
    direction[:, 0] = 1.0 / math.sqrt(n_elements)
    primitive_directional = float((lame_leaf.grad * direction).sum())
    primitive_eps = 1e2
    primitive_plus = _material_loss(
        diff,
        diff_model,
        base_lame + primitive_eps * direction,
        backend,
    )
    primitive_minus = _material_loss(
        diff,
        diff_model,
        base_lame - primitive_eps * direction,
        backend,
    )
    primitive_fd = float((primitive_plus - primitive_minus) / (2.0 * primitive_eps))

    assert math.isfinite(primitive_directional)
    assert math.isfinite(primitive_fd)
    assert primitive_directional == pytest.approx(primitive_fd, rel=1e-3, abs=1e-6)

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    lame = torch.stack(
        (
            base_lame[:, 0] * scale,
            base_lame[:, 1],
        ),
        dim=1,
    )
    loss = _material_loss(diff, diff_model, lame, backend)
    loss.backward()

    eps = 1e-4
    loss_plus = _material_loss(
        diff,
        diff_model,
        torch.stack(
            (
                base_lame[:, 0] * (float(scale.detach()) + eps),
                base_lame[:, 1],
            ),
            dim=1,
        ),
        backend,
    )
    loss_minus = _material_loss(
        diff,
        diff_model,
        torch.stack(
            (
                base_lame[:, 0] * (float(scale.detach()) - eps),
                base_lame[:, 1],
            ),
            dim=1,
        ),
        backend,
    )
    finite_difference = float((loss_plus - loss_minus) / (2.0 * eps))

    assert scale.grad is not None
    assert math.isfinite(float(scale.grad))
    assert math.isfinite(finite_difference)
    assert float(scale.grad) == pytest.approx(finite_difference, rel=1e-3, abs=1e-5)

    composition_scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    composition_lame = torch.stack(
        (
            base_lame[:, 0] * composition_scale,
            base_lame[:, 1],
        ),
        dim=1,
    )
    base_loss = _material_loss(diff, diff_model, composition_lame, backend)
    composed_loss = base_loss**2
    composed_loss.backward()
    expected_composition_grad = 2.0 * float(base_loss.detach()) * finite_difference

    assert composition_scale.grad is not None
    assert math.isfinite(float(composition_scale.grad))
    assert float(composition_scale.grad) == pytest.approx(
        expected_composition_grad,
        rel=1e-3,
        abs=1e-5,
    )
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def test_initialconditionopt_real_backend_transient_gradient_and_chain_rule(
    tmp_path,
):
    backend = pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")

    config_path = (
        ROOT
        / "polyfem-data"
        / "contact"
        / "examples"
        / "2D"
        / "initial_angular_velocity.json"
    )
    if not config_path.exists():
        pytest.skip(
            "polyfem-data submodule is not initialized; run "
            "`git submodule update --init polyfem-data`"
        )

    from polyfempy import differentiable_api as diff

    payload = _transient_elastic_initial_condition_settings(tmp_path)
    diff_model = diff.model([payload])
    n_dofs = _initial_condition_dof_count(backend, payload)
    base_u0 = torch.zeros(n_dofs, dtype=torch.float64)
    base_v0 = torch.linspace(0.0, 1.0e-2, n_dofs, dtype=torch.float64)
    base_initial_condition = torch.stack((base_u0, base_v0), dim=1)

    initial_condition_leaf = base_initial_condition.clone().requires_grad_(True)
    primitive_loss = _initial_condition_loss(
        diff,
        diff_model,
        initial_condition_leaf,
        backend,
    )
    primitive_loss.backward()

    assert tuple(primitive_loss.shape) == ()
    assert primitive_loss.dtype is torch.float64
    assert torch.isfinite(primitive_loss)
    assert initial_condition_leaf.grad is not None
    assert tuple(initial_condition_leaf.grad.shape) == (n_dofs, 2)
    assert torch.isfinite(initial_condition_leaf.grad).all()

    direction = torch.zeros_like(base_initial_condition)
    direction[-1, 1] = 1.0
    primitive_directional = float((initial_condition_leaf.grad * direction).sum())
    primitive_eps = 1e-5
    primitive_plus = _initial_condition_loss(
        diff,
        diff_model,
        base_initial_condition + primitive_eps * direction,
        backend,
    )
    primitive_minus = _initial_condition_loss(
        diff,
        diff_model,
        base_initial_condition - primitive_eps * direction,
        backend,
    )
    primitive_fd = float((primitive_plus - primitive_minus) / (2.0 * primitive_eps))

    assert math.isfinite(primitive_directional)
    assert math.isfinite(primitive_fd)
    assert primitive_directional == pytest.approx(primitive_fd, rel=1e-3, abs=1e-8)

    speed = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    initial_condition = torch.stack((base_u0, base_v0 * speed), dim=1)
    loss = _initial_condition_loss(diff, diff_model, initial_condition, backend)
    loss.backward()

    eps = 1e-5
    loss_plus = _initial_condition_loss(
        diff,
        diff_model,
        torch.stack((base_u0, base_v0 * (float(speed.detach()) + eps)), dim=1),
        backend,
    )
    loss_minus = _initial_condition_loss(
        diff,
        diff_model,
        torch.stack((base_u0, base_v0 * (float(speed.detach()) - eps)), dim=1),
        backend,
    )
    finite_difference = float((loss_plus - loss_minus) / (2.0 * eps))

    assert speed.grad is not None
    assert math.isfinite(float(speed.grad))
    assert math.isfinite(finite_difference)
    assert float(speed.grad) == pytest.approx(finite_difference, rel=1e-3, abs=1e-8)

    composition_speed = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    composition_initial_condition = torch.stack(
        (
            base_u0,
            base_v0 * composition_speed,
        ),
        dim=1,
    )
    base_loss = _initial_condition_loss(
        diff,
        diff_model,
        composition_initial_condition,
        backend,
    )
    composed_loss = base_loss**2
    composed_loss.backward()
    expected_composition_grad = 2.0 * float(base_loss.detach()) * finite_difference

    assert composition_speed.grad is not None
    assert math.isfinite(float(composition_speed.grad))
    assert float(composition_speed.grad) == pytest.approx(
        expected_composition_grad,
        rel=1e-3,
        abs=1e-8,
    )
    assert not list(tmp_path.rglob("*.vtu"))
    assert not list(tmp_path.rglob("*.pvd"))


def _initial_callable_example():
    """Keep the analytic motion checks using the current callable example."""
    import torch
    from polyfempy import differentiable_api as diff

    sample = runpy.run_path(str(ROOT / "differentiable_example/samples/initial_conditions/initial_velocity.py"))
    config = diff.model(sample["config"]).as_dicts()[0]
    config["time"].update(dt=.04, tend=.04)
    objective = diff.objectives.position(dim=0, selection=sample["body"]).final()
    obj = diff.initial(config, objective=objective)

    def loss(velocity):
        return (obj(velocity=velocity) - .02).square()

    def angle_loss(theta):
        return loss(torch.stack((theta.cos(), theta.sin())))

    return dict(loss=loss, angle_loss=angle_loss, metadata=obj.metadata,
                dof_count=obj.metadata.n_dofs, diff_model=diff.model(config),
                objective=objective)


def test_uniform_initial_velocity_and_angle_match_analytic_motion_and_fd(tmp_path, monkeypatch):
    pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    example = _initial_callable_example()
    loss = example["loss"]
    angle_loss = example["angle_loss"]
    velocity = torch.tensor([0.8, 0.6], dtype=torch.float64, requires_grad=True)
    initial = example["metadata"].uniform_initial_condition(velocity)
    assert tuple(initial.shape) == (example["dof_count"], 2)
    assert torch.equal(initial[:, 0], torch.zeros_like(initial[:, 0]))
    assert torch.equal(initial[:, 1].reshape(-1, 2), velocity.expand(initial.shape[0] // 2, 2))
    value = loss(velocity)
    value.backward()
    # Free uniform translation: x_mean(T) = vx*T on the centered unit square.
    expected_loss = (0.8 * 0.04 - 0.02) ** 2
    assert float(value.detach()) == pytest.approx(expected_loss, rel=1e-5, abs=1e-10)
    assert velocity.grad.tolist() == pytest.approx([2 * (0.8 * 0.04 - 0.02) * 0.04, 0], abs=1e-8)
    eps = 1e-4
    for component in range(2):
        direction = torch.zeros(2, dtype=torch.float64)
        direction[component] = eps
        fd = float((loss(velocity.detach() + direction) - loss(velocity.detach() - direction)).detach()) / (2 * eps)
        assert float(velocity.grad[component]) == pytest.approx(fd, rel=1e-4, abs=1e-8)

    theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    value = angle_loss(theta)
    value.backward()
    expected = -2 * (0.04 * math.cos(0.5) - 0.02) * 0.04 * math.sin(0.5)
    assert theta.grad.shape == torch.Size([])
    assert float(theta.grad) == pytest.approx(expected, rel=1e-5, abs=1e-9)
    fd = float((angle_loss(theta.detach() + eps) - angle_loss(theta.detach() - eps)).detach()) / (2 * eps)
    assert float(theta.grad) == pytest.approx(fd, rel=1e-4, abs=1e-9)


def test_body_rhs_matches_global_rhs_in_real_transient_solve(tmp_path, monkeypatch):
    pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    from polyfempy import differentiable_api as diff

    monkeypatch.chdir(tmp_path)
    example = _initial_callable_example()
    payload = example["diff_model"].as_dicts()[0]
    body_id = payload["geometry"][0]["volume_selection"]
    results = []
    for rhs in ([1.0, 2.0], [{"id": body_id, "value": [1.0, 2.0]}]):
        settings = copy.deepcopy(payload)
        settings["boundary_conditions"]["rhs"] = rhs
        velocity = torch.tensor([0.8, 0.6], dtype=torch.float64, requires_grad=True)
        value = diff.initial_condition_opt(
            diff.model([settings]), example["metadata"].uniform_initial_condition(velocity),
            objective=example["objective"],
        )()
        value.backward()
        results.append((float(value.detach()), velocity.grad.tolist()))
    # RHS follows the backend residual sign convention; check that it affects
    # motion without assuming that positive RHS means positive acceleration.
    assert abs(results[0][0] - 0.8 * 0.04) > 1e-4
    assert results[1][0] == pytest.approx(results[0][0], rel=1e-7, abs=1e-9)
    assert results[1][1] == pytest.approx(results[0][1], rel=1e-7, abs=1e-9)
