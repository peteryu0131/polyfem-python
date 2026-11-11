from pathlib import Path
import runpy
import pytest
import torch

SAMPLES = Path(__file__).resolve().parents[1] / 'differentiable_example/samples'


@pytest.mark.parametrize('kind', ['damping', 'friction', 'dirichlet', 'pressure'])
def test_small_fem_control(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(SAMPLES))
    folder = 'boundary_controls' if kind in ('dirichlet', 'pressure') else 'material_and_contact'
    sample = runpy.run_path(str(SAMPLES / folder / f'{kind}.py'), run_name="__main__")
    loss = sample['loss']
    x = sample['x'].detach().clone().requires_grad_()
    loss(x).backward()
    assert bool((x.grad.abs() > 1e-10).all())
    for h in (1e-4, 3e-5):
        differences = []
        with torch.no_grad():
            for index in range(x.numel()):
                plus, minus = x.clone(), x.clone()
                plus.reshape(-1)[index] += h
                minus.reshape(-1)[index] -= h
                differences.append((loss(plus) - loss(minus)) / (2*h))
        expected = torch.stack(differences).reshape(x.shape)
        torch.testing.assert_close(x.grad, expected, rtol=5e-3, atol=1e-8)
    with torch.no_grad():
        assert loss(sample['x']) < sample['value'].detach()

class _Session:
    def set_settings(self, config):
        import copy
        self.config = copy.deepcopy(config)

    def settings(self):
        import json
        return json.dumps(self.config)

    def model_metadata(self):
        return {'dimension': 2}

    def prepare_control(self, spec):
        self.spec = spec
        kind = spec['type']
        if kind == 'damping':
            return 2
        if kind == 'friction':
            return 1
        return len(spec['active_boundary_ids']) * (len(spec['active_time_slices']) or 2) * (2 if kind == 'dirichlet-boundary' else 1)

    def set_control_parameters(self, values):
        self.values = values.copy()

    def set_objective(self, objective):
        pass

    def solve_control_objective(self):
        return float((self.values**2).sum())

    def backward_control(self, gradient):
        return 2*self.values*gradient


def test_control_shapes_mapping_fresh_graphs_and_validation():
    from types import SimpleNamespace
    from polyfempy import differentiable_api as diff
    backend = SimpleNamespace(DifferentiableSession=_Session)
    config = {
        'time': {'time_steps': 2, 'dt': .1, 't0': 0.},
        'boundary_conditions': {'dirichlet_boundary': [
            {'id': 4, 'value': [[0., 0., 0.], [0., 0., 0.]], 'time_reference': [0., .1, .2]}]},
    }
    obj = diff.dirichlet(config, boundary_ids=[4], time_slices=[1, 0],
                                objective=diff.Objective.COMPLIANCE, backend=backend)
    assert obj.input_shape == (2, 1, 2)
    x = torch.ones(2, 1, 2, dtype=torch.float64, requires_grad=True)
    a, b = obj(values=x), obj(values=x*2)
    (a + b).backward()
    torch.testing.assert_close(x.grad, 10*x.detach())
    assert obj.spec['active_time_slices'] == [1, 0]
    with pytest.raises(ValueError, match='shape'):
        obj(values=torch.ones(4))
    with pytest.raises(ValueError):
        diff.dirichlet(config, boundary_ids=[4, 4], objective=diff.Objective.COMPLIANCE, backend=backend)
    config['boundary_conditions']['dirichlet_boundary'][0]['value'] = [0., 0.]
    with pytest.raises(ValueError, match='time series'):
        diff.dirichlet(config, boundary_ids=[4], objective=diff.Objective.COMPLIANCE, backend=backend)


def test_damping_order_and_validation():
    from types import SimpleNamespace
    from polyfempy import differentiable_api as diff
    obj = diff.damping({}, objective=diff.Objective.COMPLIANCE,
                              backend=SimpleNamespace(DifferentiableSession=_Session))
    x = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
    obj(psi=x[0], phi=x[1]).backward()
    torch.testing.assert_close(x.grad, 2*x.detach())
    with pytest.raises(ValueError):
        obj(psi=-x[0], phi=x[1])
    with pytest.raises(ValueError):
        obj(psi=x[0].float(), phi=x[1])


def test_small_fem_transient_pressure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(SAMPLES))
    from differentiable_example.samples.boundary_controls.pressure import build_scene
    from polyfempy import differentiable_api as diff
    config = diff.model(build_scene(SAMPLES.parents[1] / "differentiability-data/square.obj")).as_dicts()[0]
    config['time'] = {'dt': .05, 'tend': .1, 'integrator': {'type': 'BDF', 'steps': 1}}
    config['boundary_conditions']['pressure_boundary'][0].update(
        value=[0., 0., 0., 0.], time_reference=[0., .05, .1, .15])
    obj = diff.pressure(config, boundary_ids=[2], time_slices=[1, 0],
                               objective=diff.objectives.position(dim=0, selection=1).final())
    assert obj.input_shape == (2, 1)
    x = torch.tensor([.4, .7], dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(values=x[:, None]).square()

    loss(x).backward()
    assert bool((x.grad.abs() > 1e-10).all())
    for h in (1e-4, 3e-5):
        values = []
        with torch.no_grad():
            for i in range(2):
                plus, minus = x.clone(), x.clone()
                plus[i] += h
                minus[i] -= h
                values.append((loss(plus) - loss(minus))/(2*h))
        torch.testing.assert_close(x.grad, torch.stack(values), rtol=5e-3, atol=1e-8)


def test_friction_validation_and_chain_rule():
    from types import SimpleNamespace
    from polyfempy import differentiable_api as diff
    backend = SimpleNamespace(DifferentiableSession=_Session)
    config = {'contact': {'enabled': True}, 'solver': {'contact': {'barrier_stiffness': 100.}}}
    obj = diff.friction(config, objective=diff.Objective.COMPLIANCE, backend=backend)
    x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    obj(coefficient=x.square()).backward()
    torch.testing.assert_close(x.grad, 4*x.detach().pow(3))
    for invalid in (torch.tensor(-.1), torch.tensor(float('nan')), torch.ones(1), torch.tensor(1)):
        with pytest.raises(ValueError):
            obj(coefficient=invalid)
    config['solver']['contact']['barrier_stiffness'] = 'adaptive'
    with pytest.raises(ValueError, match='fixed'):
        diff.friction(config, objective=diff.Objective.COMPLIANCE, backend=backend)
    config['contact']['enabled'] = False
    with pytest.raises(ValueError, match='enabled contact'):
        diff.friction(config, objective=diff.Objective.COMPLIANCE, backend=backend)

def test_dirichlet_gradient_at_shared_boundary_node(tmp_path, monkeypatch):
    from differentiable_example.samples.boundary_controls.dirichlet import build_scene
    from polyfempy import differentiable_api as diff

    monkeypatch.chdir(tmp_path)
    mesh = tmp_path / "split_side.obj"
    mesh.write_text(
        "v -0.5 -0.5 0\nv 0.5 -0.5 0\nv -0.5 0 0\nv 0.5 0 0\n"
        "v -0.5 0.5 0\nv 0.5 0.5 0\nf 1 2 4\nf 1 4 3\nf 3 4 6\nf 3 6 5\n"
    )
    config = diff.model(build_scene(mesh)).as_dicts()[0]
    obj = diff.dirichlet(config, boundary_ids=[1], time_slices=[2],
                         objective=diff.objectives.position(dim=0, selection=1).final())
    x = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

    def loss(x):
        return obj(values=torch.stack((x, torch.zeros_like(x))).reshape(1, 1, 2))

    grad, = torch.autograd.grad(loss(x), x)
    with torch.no_grad():
        h = 1e-4
        fd = (loss(x + h) - loss(x - h)) / (2*h)
    assert fd.abs().item() > 1e-5
    torch.testing.assert_close(grad, fd, rtol=1e-3, atol=1e-8)
