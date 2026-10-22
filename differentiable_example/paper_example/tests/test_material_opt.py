"""Material optimization wiring checks with synthetic data; no FEM solves."""

import importlib
import ast
from pathlib import Path

import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("material_opt")


def run_example(experiment, root):
    # Execute the actual guarded script body with a temporary output directory.
    tree = ast.parse(Path(experiment.__file__).read_text())
    body = tree.body[-1]
    assert isinstance(body, ast.If)
    namespace = dict(vars(experiment), RUN_DIR=root, __name__="__main__")
    namespace["config"] = experiment.pf.config(**experiment.config.as_dict())
    namespace["config"].output.directory = str(root / "gradient")
    exec(compile(ast.Module(body=body.body, type_ignores=[]), experiment.__file__, "exec"), namespace)


def test_gradient_preserves_physics_and_disables_visualization(experiment, tmp_path):
    config = experiment.config.as_dict()
    original = importlib.import_module("impact_forward").config.as_dict()
    assert config["output"]["paraview"]["file_name"] == ""
    assert config["output"]["advanced"]["save_time_sequence"] is False
    assert config["solver"]["max_threads"] == 1
    original["solver"]["max_threads"] = 1
    original["solver"]["nonlinear"]["grad_norm_tol"] = 1e-8
    config.pop("output")
    original.pop("output")
    assert config == original
    assert config["solver"]["contact"]["barrier_stiffness"] == "adaptive"


def test_objective_selects_block_all_steps_with_unit_conversion(experiment):
    payload = experiment.objective.as_dict()
    assert payload["volume_selection"] == [2]
    assert payload["type"] == "von_mises_peak"
    assert payload["weight"] == 1e-7


def test_one_update(experiment, tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    from polyfempy.differentiable_api.model import ModelMetadata

    metadata = ModelMetadata(2, 6, 2, 6, 2, 12, False, (1, 2))
    calls = []
    backwards = []
    configurations = []
    parameters = []

    class Model:
        def metadata(self):
            return metadata

    class Problem:
        diagnostics = {"step": 12}

        def __init__(self, materials):
            assert len(materials) == 1
            body, fields = next(iter(materials.items()))
            from polyfempy.differentiable_api.objectives import _selection_id
            assert _selection_id(body) == 1
            assert fields["unit"] == "MPa" and set(fields) == {"E", "unit"}
            self.young = fields["E"]
            parameters.append(self.young)

        def optimizer(self, optimizer, **kwargs):
            return experiment.diff.optimization(self, parameters=[self.young]).optimizer(optimizer, **kwargs)

        def __call__(self):
            calls.append(float(self.young.detach()))
            value = (self.young - 19.)**2
            value.register_hook(lambda grad: backwards.append(float(grad)))
            return value

    def build_model(config):
        configurations.append(config.as_dict())
        return Model()

    monkeypatch.setattr(experiment.diff, "model", build_model)
    monkeypatch.setattr(experiment.diff, "material_opt", lambda model, *, materials, objective: Problem(materials))
    root = tmp_path / "run"
    run_example(experiment, root)
    assert calls == pytest.approx([20.])
    assert backwards == [1.]
    assert len(parameters) == 1
    assert float(parameters[0].detach()) == pytest.approx(19.98)
    assert float(parameters[0].grad) == pytest.approx(2.)
    assert capsys.readouterr().out.strip() == "Updated lattice E: 19.980000 MPa"
    assert not (root / "result.json").exists()
    assert len(configurations) == 1
    with pytest.raises(FileExistsError):
        run_example(experiment, root)
    assert len(calls) == 1
