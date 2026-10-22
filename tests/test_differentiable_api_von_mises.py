"""Small static stress checks; no impact simulation or visualization output.

Requires rebuilding the native backend. Finite differences deliberately solve
again, but each model has only two bodies with four elements each.
"""

import numpy as np
import pytest
import torch

from polyfempy import differentiable_api as diff
from polyfempy.generated_api import generated_api as pf


def test_body_material_mapping_preserves_element_order_and_gradients():
    metadata = diff.ModelMetadata(2, 5, 4, 5, 2, 10, True, [2, 0, 2, 0])
    first = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([5., 7.], dtype=torch.float64, requires_grad=True)
    # Input dictionary order must not determine element order.
    mapped = metadata.material_by_body({0: first, 2: second})
    torch.testing.assert_close(mapped, torch.stack([second, first, second, first]))
    mapped.square().sum().backward()
    torch.testing.assert_close(first.grad, 4 * first.detach())
    torch.testing.assert_close(second.grad, 4 * second.detach())
    with pytest.raises(ValueError, match="every body"):
        metadata.material_by_body({0: first})
    with pytest.raises(ValueError, match="same dtype"):
        metadata.material_by_body({0: first, 2: second.float()})
    with pytest.raises(RuntimeError, match="rebuild"):
        diff.ModelMetadata(2, 5, 4, 5, 2, 10, True).material_by_body({0: first})


def test_von_mises_objective_payload():
    assert diff.objectives.von_mises_squared(selection=2, weight=2.5).as_dict() == {
        "type": "von_mises_squared", "state": 0, "volume_selection": [2],
        "weight": 2.5, "print_energy": "",
    }
    final = diff.objectives.von_mises_squared(selection=2, weight=2.5).final().as_dict()
    assert final["type"] == "transient_integral"
    assert final["integral_type"] == "final"
    assert final["weight"] == 1.0
    assert final["static_objective"]["weight"] == 2.5
    assert final["static_objective"]["volume_selection"] == [2]


def _mesh(path, dimension, shift):
    # One unconstrained interior node permits a nonzero adjoint contribution.
    if dimension == 2:
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [.5, .5, 0.]])
        elements = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        kind = 2
    else:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.25, .25, .25]])
        elements = [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
        kind = 4
    for element in elements:
        edges = points[element[1:], :dimension] - points[element[0], :dimension]
        if np.linalg.det(edges) < 0:
            element[1], element[2] = element[2], element[1]
    points[:, 0] += shift
    nodes = [f"{i} {x} {y} {z}" for i, (x, y, z) in enumerate(points, 1)]
    cells = [f"{i} {kind} 0 " + " ".join(str(v + 1) for v in e)
             for i, e in enumerate(elements, 1)]
    path.write_text("\n".join([
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", "5", *nodes,
        "$EndNodes", "$Elements", "4", *cells, "$EndElements", "",
    ]), encoding="utf-8")


def _model(tmp_path, dimension, formulation, loaded):
    model = pf.model()
    bodies = []
    for index in range(2):
        mesh = tmp_path / f"body{index}.msh"
        _mesh(mesh, dimension, shift=3 * index)
        body = pf.body(model=model)
        body.mesh(pf.mesh(mesh=str(mesh)))
        factory = pf.neo_hookean if formulation == "NeoHookean" else pf.linear_elasticity
        body.material(factory(E=100., nu=.25))
        prescribed = ["0.03*x+0.01*y", "-0.01*y", "0.005*z"][:dimension]
        body.surface_all(id=index + 1).dirichlet(value=prescribed)
        body.rhs(value=([20., 10., 5.] if loaded else [0., 0., 0.])[:dimension])
        bodies.append(body)
    config = pf.config(
        model=model,
        space=pf.space(discr_order=1, advanced=pf.space_advanced(quadrature_order=2)),
        solver=pf.solver(
            max_threads=1, linear=pf.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=pf.nonlinear(norm_type="Euclidean", rel_grad_norm_tol=0,
                                   first_grad_norm_tol=1e-12, grad_norm_tol=1e-10),
            advanced=pf.solver_advanced(characteristic_force_density=1, characteristic_length=1),
        ),
        output=pf.output(directory=str(tmp_path), json="",
                         paraview=pf.output_paraview(file_name=""),
                         advanced=pf.output_advanced(save_time_sequence=False)),
    )
    return diff.model(config), bodies


def _affine_value(dimension, formulation, E):
    gradient = np.diag([.03, -.01, .005][:dimension])
    gradient[0, 1] = .01
    lam, mu = .4 * E, .4 * E  # nu=0.25
    if formulation == "LinearElasticity":
        stress = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(dimension)
    else:
        F = np.eye(dimension) + gradient
        J = np.linalg.det(F)
        stress = (mu * (F @ F.T - np.eye(dimension)) + lam * np.log(J) * np.eye(dimension)) / J
    # Independent scalar formula, including the backend's 2D convention.
    if dimension == 2:
        vm2 = stress[0, 0]**2 - stress[0, 0]*stress[1, 1] + stress[1, 1]**2 + 3*stress[0, 1]**2
    else:
        vm2 = sum((stress[i, i] - stress[j, j])**2 / 2 + 3*stress[i, j]**2
                  for i, j in [(0, 1), (1, 2), (0, 2)])
    return vm2 * (1 if dimension == 2 else 1/6)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("formulation", ["LinearElasticity", "NeoHookean"])
@pytest.mark.parametrize("loaded", [False, True], ids=["prescribed-strain", "with-body-force"])
def test_static_von_mises_value_and_full_material_gradient(tmp_path, dimension, formulation, loaded):
    model, bodies = _model(tmp_path, dimension, formulation, loaded)
    metadata = model.metadata()
    assert len(metadata.element_body_ids) == 8, "Rebuild the native backend first"
    weight = 1.7

    def loss(E):
        pairs = [torch.stack([.4 * value, .4 * value]) for value in E]
        lame = metadata.material_by_body(dict(zip(bodies, pairs)))
        return diff.material_opt(model, lame, objective=diff.objectives.von_mises_squared(
            selection=bodies[0], weight=weight))()

    E = torch.tensor([100., 150.], dtype=torch.float64, requires_grad=True)
    value = loss(E)
    value.backward()
    assert torch.isfinite(value) and float(value.detach()) > 0
    assert torch.isfinite(E.grad).all()
    assert float(E.grad[0]) > 0
    # The other, disconnected body's material cannot affect this objective.
    assert float(E.grad[1]) == pytest.approx(0, abs=1e-9)
    for h in (.1, .01):
        perturbation = torch.tensor([h, 0.], dtype=torch.float64)
        fd = (loss(E.detach() + perturbation) - loss(E.detach() - perturbation)) / (2 * h)
        assert float(E.grad[0]) == pytest.approx(float(fd.detach()), rel=2e-4, abs=1e-7)
    if not loaded:
        expected = weight * _affine_value(dimension, formulation, 100.)
        assert float(value.detach()) == pytest.approx(expected, rel=1e-7, abs=1e-9)
        assert float(E.grad[0]) == pytest.approx(2 * expected / 100., rel=1e-6)
    else:
        # At fixed displacement stress scales with E. A different total
        # derivative demonstrates that displacement sensitivity contributes.
        direct_only = 2 * float(value.detach()) / 100.
        assert abs(float(E.grad[0]) - direct_only) > 1e-5
    assert not list(tmp_path.rglob("*.vtu"))


def test_unknown_body_selection_is_rejected(tmp_path):
    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    metadata = model.metadata()
    lame = metadata.uniform_material(torch.tensor([40., 40.], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="unknown body ID"):
        diff.material_opt(model, lame, objective=diff.objectives.von_mises_squared(selection=99))()


def test_unwrapped_transient_material_objective_is_rejected(tmp_path):
    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    metadata = model.metadata()
    settings = model.as_dicts()[0]
    settings["time"] = {"dt": .1, "tend": .2}
    lame = metadata.uniform_material(torch.tensor([40., 40.], dtype=torch.float64))
    with pytest.raises(RuntimeError, match=r"require .final\(\)"):
        diff.material_opt(diff.model(settings), lame,
                          objective=diff.objectives.von_mises_squared())()


def test_nonmaterial_session_is_rejected(tmp_path):
    from polyfempy import polyfempy as backend

    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    session = backend.DifferentiableSession()
    session.set_settings(model.as_dicts()[0])
    with pytest.raises(RuntimeError, match="material_opt only"):
        session.set_objective(diff.objectives.von_mises_squared().as_dict())


@pytest.mark.parametrize("invalid", ["static-final", "time-average", "other-state"])
def test_final_von_mises_rejects_unsupported_combinations(tmp_path, invalid):
    from polyfempy import polyfempy as backend

    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    settings = model.as_dicts()[0]
    if invalid != "static-final":
        settings["time"] = {"dt": .1, "tend": .2}
    session = backend.DifferentiableSession()
    session.set_settings(settings)
    metadata = session.model_metadata()
    session.set_material_lame_parameters(np.full(2 * metadata["n_elements"], 40.))
    objective = diff.objectives.von_mises_squared().final().as_dict()
    if invalid == "time-average":
        objective["integral_type"] = "trapezoidal"
    if invalid == "other-state":
        objective["static_objective"]["state"] = 1
    message = {"static-final": "static models", "time-average": "only final-time", "other-state": "state=0"}[invalid]
    with pytest.raises(RuntimeError, match=message):
        session.set_objective(objective)
