import copy
from pathlib import Path
import runpy

import pytest
import torch

from polyfempy import differentiable_api as diff
from test_differentiable_api_material_enu import _backend, _payload


def _configuration(E):
    import ast
    path = Path(__file__).resolve().parents[1] / 'differentiable_example/samples/material_and_contact/material_matching.py'
    tree = ast.parse(path.read_text())
    end = next(i for i, node in enumerate(tree.body)
               if isinstance(node, ast.FunctionDef) and node.name == 'configuration')
    tree.body = tree.body[:end + 1]
    namespace = {'__file__': str(path)}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['configuration'](E)


def test_target_config_is_copied_and_invalid_objectives_fail_before_setup():
    target = _payload()
    obj = diff.material(_payload(), body=1, target=target,
                               objective=diff.objectives.target(target_state=1), backend=_backend())
    target['materials'][0]['nu'] = .4
    assert obj._target['materials'][0]['nu'] == .25
    for objective in (diff.Objective.COMPLIANCE, diff.objectives.target(target_state=0),
                      diff.objectives.target(state=1, target_state=1)):
        with pytest.raises(ValueError, match='target requires'):
            diff.material({}, body=1, target={}, objective=objective)
    with pytest.raises(ValueError, match='requires target'):
        diff.material({}, body=1, objective=diff.objectives.target(target_state=1))


def test_old_backend_target_has_clear_error():
    from polyfempy.differentiable_api._backend import BackendContractError
    obj = diff.material(_payload(), body=1, target=_payload(), backend=_backend(),
                               objective=diff.objectives.target(target_state=1))
    with pytest.raises(BackendContractError, match='Rebuild.*set_target_settings'):
        obj(E=torch.tensor(100., dtype=torch.float64))


def test_small_fem_material_matching_sample_and_gradient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / 'differentiable_example/samples/material_and_contact/material_matching.py'
    sample = runpy.run_path(str(path))
    assert sample['x'].item() > 1.
    obj = sample['obj']
    x = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)

    def loss(x):
        return (obj(E=100*x.square()) * 1e5).square()

    value = loss(x)
    value.backward()
    assert x.grad < 0 and value > 0
    with torch.no_grad():
        for h in (1e-4, 3e-5):
            fd = (loss(x+h) - loss(x-h)) / (2*h)
            torch.testing.assert_close(x.grad, fd, rtol=2e-3, atol=1e-7)
        assert sample['loss'](sample['x']) < sample['value'].detach()
        assert obj(E=torch.tensor(200., dtype=torch.float64)).item() == pytest.approx(0., abs=1e-12)


def test_small_fem_target_rejects_mismatched_mesh_and_state_indices(tmp_path, monkeypatch):
    from polyfempy import polyfempy as native
    monkeypatch.chdir(tmp_path)
    config, _ = _configuration(100.)
    payload = config.as_dict()
    session = native.DifferentiableSession()
    session.set_settings(payload)
    with pytest.raises(RuntimeError, match='target_state=1 requires'):
        session.set_objective(diff.objectives.target(target_state=1).as_dict())
    session.set_settings(payload)
    with pytest.raises(RuntimeError, match='state=0 only'):
        session.set_objective(diff.objectives.compliance(state=2).as_dict())
    session.set_settings(payload)
    bad = copy.deepcopy(payload)
    bad['geometry'][0]['transformation'] = {'scale': [2., 1.]}
    with pytest.raises(RuntimeError, match='identical reference vertices'):
        session.set_target_settings(bad)


def test_small_fem_transient_material_target_gradient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, body = _configuration(10.)
    payload = config.as_dict()
    payload['materials'][0].update(type='NeoHookean', rho=1.)
    payload['time'] = {'t0': 0., 'dt': .05, 'tend': .1,
                       'integrator': {'type': 'BDF', 'steps': 1}}
    payload['solver']['nonlinear'] = {'grad_norm_tol': 1e-10}
    target = copy.deepcopy(payload)
    target['materials'][0]['E'] = 20.
    obj = diff.material(
        payload, body=body, target=target,
        objective=diff.objectives.transient_integral(
            static_objective=diff.objectives.target(target_state=1), integral_type='uniform'),
    )
    E = torch.tensor(12., dtype=torch.float64, requires_grad=True)
    obj(E=E).backward()
    assert E.grad.abs() > 1e-12
    with torch.no_grad():
        h = 1e-3
        fd = (obj(E=E+h) - obj(E=E-h)) / (2*h)
        torch.testing.assert_close(E.grad, fd, rtol=3e-3, atol=1e-10)
        assert obj(E=E.new_tensor(20.)).item() == pytest.approx(0., abs=1e-12)
