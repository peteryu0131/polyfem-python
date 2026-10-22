"""Small native peak-objective checks; never runs the paper lattice."""
import json
import numpy as np
import pytest
import torch
from polyfempy import differentiable_api as diff
from test_differentiable_api_von_mises import _model, _affine_value
from test_differentiable_api_contact_material import _configuration


def test_peak_payload_and_no_final_wrapper():
    objective = diff.objectives.von_mises_peak(selection=2, weight=1e-7)
    assert objective.as_dict()["type"] == "von_mises_peak"
    assert objective.as_dict()["volume_selection"] == [2]
    with pytest.raises(ValueError, match="spatial objective"):
        objective.final()


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("formulation", ["LinearElasticity", "NeoHookean"])
@pytest.mark.parametrize("loaded", [False, True])
def test_static_peak_value_and_material_derivative(tmp_path, dimension, formulation, loaded):
    model, bodies = _model(tmp_path, dimension, formulation, loaded)
    metadata = model.metadata()
    def evaluate(E):
        pairs = [torch.stack([.4 * value, .4 * value]) for value in E]
        problem = diff.material_opt(model, metadata.material_by_body(dict(zip(bodies, pairs))),
                                   objective=diff.objectives.von_mises_peak(selection=bodies[0], weight=1.7))
        return problem(), problem.diagnostics
    E = torch.tensor([100., 150.], dtype=torch.float64, requires_grad=True)
    value, info = evaluate(E)
    value.backward()
    assert info["step"] == 0
    assert info["value"] == pytest.approx(float(value.detach()))
    assert float(E.grad[1]) == pytest.approx(0., abs=1e-9)
    for h in (.1, .01):
        delta = torch.tensor([h, 0.], dtype=torch.float64)
        plus, _ = evaluate(E.detach() + delta)
        minus, _ = evaluate(E.detach() - delta)
        fd = float((plus.detach() - minus.detach()) / (2*h))
        assert float(E.grad[0]) == pytest.approx(fd, rel=3e-4, abs=1e-7)
    if not loaded:
        volume = 1. if dimension == 2 else 1./6
        expected = 1.7 * np.sqrt(_affine_value(dimension, formulation, 100.) / volume)
        assert float(value.detach()) == pytest.approx(expected, rel=1e-7)
        assert float(E.grad[0]) == pytest.approx(expected/100, rel=1e-6)


def test_peak_contact_cross_body_gradient_and_independent_history(tmp_path):
    from polyfempy import solve
    from polyfempy.generated_api import generated_api as pf
    import meshio
    config, lower, upper = _configuration(tmp_path)
    model = diff.model(config)
    metadata = model.metadata()
    def evaluate(E):
        lower_lame = torch.stack([E*.25/(1-.25**2), E/2.5])
        fixed_upper = E.new_tensor([500*.25/(1-.25**2), 500/2.5])
        problem = diff.material_opt(model, metadata.material_by_body({lower: lower_lame, upper: fixed_upper}),
                                   objective=diff.objectives.von_mises_peak(selection=upper, weight=1.7))
        return problem(), problem.diagnostics
    E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    value, info = evaluate(E)
    value.backward()
    assert abs(float(E.grad)) > 1e-9
    for h in (.1, .03):
        plus, _ = evaluate(E.detach()+h)
        minus, _ = evaluate(E.detach()-h)
        fd = float((plus.detach()-minus.detach())/(2*h))
        assert float(E.grad) == pytest.approx(fd, rel=1e-2, abs=1e-9)
    config.output.paraview = pf.output_paraview(file_name="peak.pvd", skip_frame=1,
        surface=False, wireframe=False, points=False, vismesh_rel_area=10000000,
        options=pf.options(body_ids=True, scalar_values=True, tensor_values=False))
    config.output.advanced = pf.output_advanced(save_time_sequence=True, save_solve_sequence_debug=False)
    solve(cfg=config)
    history = []
    for step in range(21):
        mesh = meshio.read(tmp_path / f"step_{step}.vtu")
        ids = mesh.point_data["body_ids"].ravel()
        vm = mesh.point_data["von_mises"].ravel()
        history.append(1.7 * float(vm[ids == 2].max()))
    assert float(value.detach()) == pytest.approx(max(history), rel=1e-7, abs=1e-9)
    assert history[info["step"]] == pytest.approx(max(history), rel=1e-7)
    print(json.dumps({"peak": info, "gradient": float(E.grad), "history": history}))


def test_transient_peak_precedes_final_step(tmp_path):
    # A short prescribed strain pulse; low mass makes the interior nearly affine.
    model, bodies = _model(tmp_path, 2, "NeoHookean", False)
    payload = model.as_dicts()[0]
    payload["time"] = {"dt": .02, "tend": .08, "integrator": "BDF"}
    for material in payload["materials"]:
        material["rho"] = 1e-8
    for boundary in payload["boundary_conditions"]["dirichlet_boundary"]:
        boundary["value"] = [f"({v})*t*(0.08-t)/0.0016" for v in boundary["value"]]
    model = diff.model(payload)
    metadata = model.metadata()
    E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    problem = diff.material_opt(model, metadata.uniform_material(torch.stack([.4*E,.4*E])),
                               objective=diff.objectives.von_mises_peak(selection=bodies[0]))
    value = problem()
    value.backward()
    assert problem.diagnostics["step"] == 2
    assert problem.diagnostics["time"] == pytest.approx(.04)
    assert float(value.detach()) == pytest.approx(np.sqrt(_affine_value(2,"NeoHookean",100.)), rel=1e-5)
    assert float(E.grad) == pytest.approx(float(value.detach())/100, rel=1e-5)


@pytest.mark.parametrize("kind", ["unknown-body", "other-state"])
def test_peak_rejects_unsupported_inputs_before_solve(tmp_path, kind):
    from polyfempy import polyfempy as backend
    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    session = backend.DifferentiableSession()
    session.set_settings(model.as_dicts()[0])
    session.set_material_lame_parameters(np.full(2*session.model_metadata()["n_elements"], 40.))
    objective = diff.objectives.von_mises_peak(selection=99 if kind=="unknown-body" else 1,
                                             state=1 if kind=="other-state" else 0)
    message = {"unknown-body":"unknown body", "other-state":"state=0"}[kind]
    with pytest.raises(RuntimeError, match=message):
        session.set_objective(objective.as_dict())


def test_zero_peak_uses_zero_subgradient_and_first_sample(tmp_path):
    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    payload = model.as_dicts()[0]
    for boundary in payload["boundary_conditions"]["dirichlet_boundary"]:
        boundary["value"] = [0., 0.]
    model = diff.model(payload)
    metadata = model.metadata()
    E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    problem = diff.material_opt(model, metadata.uniform_material(torch.stack([.4*E,.4*E])),
                               objective=diff.objectives.von_mises_peak())
    value = problem()
    value.backward()
    assert float(value.detach()) == 0.
    assert float(E.grad) == 0.
    assert problem.diagnostics["step"] == 0
    assert problem.diagnostics["element"] == 0
    assert problem.diagnostics["quadrature_point"] == 0


def test_linear_transient_peak_is_rejected_before_solve(tmp_path):
    from polyfempy import polyfempy as backend
    model, _ = _model(tmp_path, 2, "LinearElasticity", False)
    payload = model.as_dicts()[0]
    payload["time"] = {"dt": .02, "tend": .08}
    session = backend.DifferentiableSession()
    session.set_settings(payload)
    session.set_material_lame_parameters(np.full(2*session.model_metadata()["n_elements"],40.))
    with pytest.raises(RuntimeError, match="require NeoHookean"):
        session.set_objective(diff.objectives.von_mises_peak().as_dict())
