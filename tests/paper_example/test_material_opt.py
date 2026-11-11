"""Material optimization wiring checks with synthetic data; no FEM solves."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PAPER_EXAMPLE = Path(__file__).resolve().parents[2] / "differentiable_example" / "paper_example"

@pytest.fixture
def experiment(monkeypatch):
    path = PAPER_EXAMPLE / "material_opt.py"
    monkeypatch.syspath_prepend(str(path.parent))
    tree = ast.parse(path.read_text())
    # Load configuration without creating an output folder or starting a solve.
    split = next(
        i + 1 for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "config"
                for target in node.targets)
    )
    namespace = {"__file__": str(path), "__name__": "material_opt_test"}
    exec(compile(ast.Module(body=tree.body[:split], type_ignores=[]), str(path), "exec"), namespace)
    namespace["execution_body"] = tree.body[split:]
    return SimpleNamespace(**namespace)


def run_example(experiment, root):
    # Execute the actual optimization statements with a temporary output directory.
    namespace = dict(vars(experiment), OUTPUT_DIR=root)
    namespace["config"] = experiment.polyfem.config(**experiment.config.as_dict())
    namespace["config"].output.directory = str(root)
    exec(compile(ast.Module(body=experiment.execution_body, type_ignores=[]),
                 experiment.__file__, "exec"), namespace)
    return namespace


def test_gradient_preserves_physics_and_disables_visualization(experiment, tmp_path):
    config = experiment.config.as_dict()
    from test_impact_forward import load_example

    original = load_example()["config"].as_dict()
    assert config["output"]["paraview"]["file_name"] == ""
    assert config["output"]["advanced"]["save_time_sequence"] is False
    assert config["solver"]["max_threads"] == 1
    original["solver"]["max_threads"] = 1
    original["solver"]["nonlinear"]["grad_norm_tol"] = 1e-8
    config.pop("output")
    original.pop("output")
    assert config == original
    assert config["solver"]["contact"]["barrier_stiffness"] == "adaptive"


def test_one_update(experiment, tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    calls = []
    backwards = []
    configurations = []

    class Output:
        n_elements = 3

        def __call__(self, *, E, nu):
            assert E.shape == nu.shape == (self.n_elements,)
            calls.append((E.detach().tolist(), nu.detach().tolist()))
            value = ((E / 1e7 - 19.)**2 + (nu - .3)**2).mean() * 1e7
            value.register_hook(lambda grad: backwards.append(float(grad)))
            return value

    def build_object(config, *, body, selection, objective):
        from polyfempy.differentiable_api.objectives import _selection_id
        assert _selection_id(body) == 1
        assert _selection_id(selection) == 2
        assert objective == experiment.diff.Objective.VON_MISES_PEAK
        configurations.append(config.as_dict())
        return Output()

    monkeypatch.setattr(experiment.diff, "material", build_object)
    root = tmp_path / "run"
    result = run_example(experiment, root)
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx([20e7] * 3)
    assert calls[0][1] == pytest.approx([.45] * 3)
    assert backwards == pytest.approx([1e-7])
    assert result["x"].detach().tolist() == pytest.approx([19.999, .449])
    assert result["x"].grad.tolist() == pytest.approx([2., .3])
    assert capsys.readouterr().out.strip().splitlines() == [
        "Updated lattice E: 19.999000 MPa", "Updated lattice nu: 0.449000"]
    assert not (root / "result.json").exists()
    assert len(configurations) == 1
    with pytest.raises(FileExistsError):
        run_example(experiment, root)
    assert len(calls) == 1
