"""Shared parameter preparation, including real solves on a replacement mesh."""

from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from polyfempy import differentiable_api as diff


def test_metadata_queries_backend_and_does_not_share_settings():
    class Session:
        def set_settings(self, settings):
            settings["mesh"] = "mutated"

        def model_metadata(self):
            return dict(dimension=2, n_vertices=5, n_elements=4, n_bases=5,
                        value_dim=2, n_dofs=10, supports_uniform_initial_condition=True)

    model = diff.model({"mesh": "original"})
    metadata = model.metadata(backend=SimpleNamespace(DifferentiableSession=Session))
    assert metadata.n_elements == 4
    assert metadata.n_dofs == 10
    assert model.as_dicts() == [{"mesh": "original"}]
    with pytest.raises(RuntimeError, match="rebuild"):
        model.metadata(backend=SimpleNamespace())
    with pytest.raises(ValueError, match="exactly one"):
        diff.model([{}, {}]).metadata(backend=SimpleNamespace())


def test_uniform_preparation_preserves_chain_rule_and_checks_inputs():
    torch = pytest.importorskip("torch")
    metadata = diff.ModelMetadata(2, 5, 4, 5, 2, 10, True)
    velocity = torch.tensor([0.8, 0.6], dtype=torch.float64, requires_grad=True)
    displacement = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    initial = metadata.uniform_initial_condition(velocity, displacement=displacement)
    assert initial.shape == (10, 2)
    initial.square().sum().backward()
    torch.testing.assert_close(velocity.grad, 10 * velocity.detach())
    torch.testing.assert_close(displacement.grad, 10 * displacement.detach())
    lame = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    material = metadata.uniform_material(lame)
    assert material.shape == (4, 2)
    material.square().sum().backward()
    torch.testing.assert_close(lame.grad, 8 * lame.detach())
    with pytest.raises(ValueError, match="unconstrained P1"):
        replace(metadata, supports_uniform_initial_condition=False).uniform_initial_condition(velocity)
    with pytest.raises(TypeError, match="torch.Tensor"):
        metadata.uniform_material([2.0, 3.0])
    for bad in (torch.ones(3), torch.ones(2, dtype=torch.int64), torch.tensor([float("nan"), 1.0])):
        with pytest.raises(ValueError):
            metadata.uniform_initial_condition(bad)
        with pytest.raises(ValueError):
            metadata.uniform_material(bad)
    with pytest.raises(ValueError, match="same dtype"):
        metadata.uniform_initial_condition(velocity, displacement=displacement.float())


def test_preparation_errors_identify_actual_input_and_correction():
    torch = pytest.importorskip("torch")
    metadata = diff.ModelMetadata(3, 4, 1, 4, 3, 12, True)
    with pytest.raises(ValueError, match=r"shape \(3,\).*got \(2,\)"):
        metadata.uniform_initial_condition(torch.zeros(2))
    with pytest.raises(ValueError, match=r"\[lambda, mu\].*got \(4, 2\)"):
        metadata.uniform_material(torch.zeros(4, 2))
    with pytest.raises(ValueError, match=r"torch.int64.*to\(dtype=torch.float64\)"):
        metadata.uniform_material(torch.tensor([1, 2]))


def test_shared_and_independent_material_parameters_support_optimizer_loops():
    torch = pytest.importorskip("torch")
    metadata = diff.ModelMetadata(2, 5, 4, 5, 2, 10, True)
    pair = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    shared_optimizer = torch.optim.SGD([pair], lr=0.01)
    independent = metadata.uniform_material(pair.detach()).requires_grad_(True)
    independent_optimizer = torch.optim.SGD([independent], lr=0.01)
    weights = torch.arange(1, 5, dtype=torch.float64).unsqueeze(1)
    for _ in range(2):
        shared_optimizer.zero_grad()
        broadcast = metadata.uniform_material(pair)
        (weights * broadcast.square()).sum().backward()
        torch.testing.assert_close(pair.grad, 2 * weights.sum() * pair.detach())
        shared_optimizer.step()
        independent_optimizer.zero_grad()
        (weights * independent.square()).sum().backward()
        torch.testing.assert_close(independent.grad, 2 * weights * independent.detach())
        independent_optimizer.step()
    assert independent.is_leaf
    assert not torch.equal(independent[0], independent[-1])


@pytest.fixture
def replacement_mesh(tmp_path):
    # Same centered unit square, different topology: 5 vertices / 4 triangles.
    # Gmsh format also ensures material setup no longer depends on OBJ parsing.
    mesh = tmp_path / "square_center.msh"
    mesh.write_text("""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 -0.5 -0.5 0
2 0.5 -0.5 0
3 0.5 0.5 0
4 -0.5 0.5 0
5 0 0 0
$EndNodes
$Elements
4
1 2 0 1 2 5
2 2 0 2 3 5
3 2 0 3 4 5
4 2 0 4 1 5
$EndElements
""", encoding="utf-8")
    return mesh


@pytest.mark.parametrize("threads", [None, 0, 1, 4])
def test_session_thread_settings(replacement_mesh, tmp_path, threads):
    backend = pytest.importorskip("polyfempy.polyfempy")
    session = backend.DifferentiableSession()
    config = {
        "geometry": [{"mesh": str(replacement_mesh)}],
        "materials": {"type": "LinearElasticity", "E": 100., "nu": .3, "rho": 1.},
        "space": {"discr_order": 1},
        "contact": {"enabled": False},
        "output": {"directory": str(tmp_path), "json": "",
                   "paraview": {"file_name": ""}},
    }
    if threads is not None:
        config["solver"] = {"max_threads": threads}
    # Only mesh/FE preparation: no forward or adjoint solve.
    session.set_settings(config)
    assert json.loads(session.settings())["solver"]["max_threads"] == (0 if threads is None else threads)
    # Reusing a session with an omitted setting must reset to the default.
    config.pop("solver", None)
    session.set_settings(config)
    assert json.loads(session.settings())["solver"]["max_threads"] == 0


@pytest.mark.parametrize("threads", [-1, 1.5, True, "4"])
def test_session_rejects_invalid_thread_settings(threads):
    backend = pytest.importorskip("polyfempy.polyfempy")
    session = backend.DifferentiableSession()
    with pytest.raises(RuntimeError, match="non-negative integer"):
        session.set_settings({"solver": {"max_threads": threads}})


def test_initial_condition_example_on_replacement_mesh(replacement_mesh, tmp_path, monkeypatch):
    pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    from test_differentiable_api_real_backend import _initial_callable_example
    monkeypatch.chdir(tmp_path)
    example = _initial_callable_example()
    settings = example["diff_model"].as_dicts()[0]
    settings["geometry"][0]["mesh"] = str(replacement_mesh)
    settings["output"]["directory"] = str(tmp_path / "initial")
    model = diff.model(settings)
    metadata = model.metadata()
    assert (metadata.n_vertices, metadata.n_elements, metadata.n_dofs) == (5, 4, 10)
    assert metadata.n_dofs != example["metadata"].n_dofs
    obj = diff.initial(model, objective=example["objective"])

    def loss(velocity):
        return (obj(velocity=velocity) - .02).square()
    velocity = torch.tensor([0.8, 0.6], dtype=torch.float64, requires_grad=True)
    value = loss(velocity)
    value.backward()
    assert float(value.detach()) == pytest.approx((0.8 * 0.04 - 0.02) ** 2, abs=1e-9)
    assert velocity.grad.tolist() == pytest.approx([2 * (0.8 * 0.04 - 0.02) * 0.04, 0], abs=1e-8)
    step = torch.tensor([1e-4, 0], dtype=torch.float64)
    fd = (loss(velocity.detach() + step) - loss(velocity.detach() - step)) / 2e-4
    assert float(velocity.grad[0]) == pytest.approx(float(fd.detach()), rel=1e-4)

    # Higher order has more basis functions than vertices; never silently repeat
    # a vertex-count-based vector into this different layout.
    settings["space"]["discr_order"] = 2
    higher_order = diff.model(settings).metadata()
    assert higher_order.n_bases > higher_order.n_vertices
    with pytest.raises(ValueError, match="unconstrained P1"):
        higher_order.uniform_initial_condition(velocity)


def test_material_example_on_replacement_mesh(replacement_mesh, tmp_path, monkeypatch):
    pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    example = runpy.run_path(str(root / "differentiable_example/samples/material_and_contact/material_field.py"))
    assert example["x"].is_leaf
    assert example["x"].grad.shape == example["x"].shape
    settings = diff.model(example["config"]).as_dicts()[0]
    settings["geometry"][0]["mesh"] = str(replacement_mesh)
    settings["geometry"][0]["surface_selection"][0]["box"] = [[-0.501, -0.501], [-0.499, 0.501]]
    settings["output"]["directory"] = str(tmp_path / "material")
    model = diff.model(settings)
    metadata = model.metadata()
    assert (metadata.n_elements, metadata.n_dofs) == (4, 10)
    assert metadata.n_elements != example["obj"].n_elements
    with pytest.raises(ValueError, match="unconstrained P1"):
        metadata.uniform_initial_condition(torch.zeros(2))

    # A single material pair shared by all elements is also differentiable.
    def loss(scale):
        pair = scale * torch.tensor([1e5, 5e4], dtype=torch.float64)
        return diff.material_opt(model, metadata.uniform_material(pair),
                                 objective=diff.Objective.COMPLIANCE,
                                 objective_params={"selection": 1})()

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    value = loss(scale)
    value.backward()
    assert torch.isfinite(value) and abs(float(value.detach())) > 1e-12
    fd = (loss(scale.detach() + 1e-4) - loss(scale.detach() - 1e-4)) / 2e-4
    assert float(scale.grad) == pytest.approx(float(fd.detach()), rel=1e-3, abs=1e-9)
