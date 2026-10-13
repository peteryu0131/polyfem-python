"""Shared parameter preparation, including real solves on a replacement mesh."""

from dataclasses import replace
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


def test_initial_condition_example_on_replacement_mesh(replacement_mesh, tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    example = runpy.run_path(str(root / "differentiable_example/initial_condition/transient_elastic_initial_condition.py"))
    settings = example["diff_model"].as_dicts()[0]
    settings["geometry"][0]["mesh"] = str(replacement_mesh)
    settings["output"]["directory"] = str(tmp_path / "initial")
    model = diff.model(settings)
    metadata = model.metadata()
    assert (metadata.n_vertices, metadata.n_elements, metadata.n_dofs) == (5, 4, 10)
    assert metadata.n_dofs != example["metadata"].n_dofs
    # Exercise the real example's loss, replacing only model + its metadata.
    loss = example["loss"]
    loss.__globals__.update(diff_model=model, metadata=metadata)
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


def test_material_example_on_replacement_mesh(replacement_mesh, tmp_path):
    pytest.importorskip("polyfempy.polyfempy")
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    # Runs the original bar example too, preserving per-element leaf gradients.
    example = runpy.run_path(str(root / "differentiable_example/material/linear_elasticity_compliance.py"))
    assert example["lame"].is_leaf
    assert example["lame"].grad.shape == example["lame"].shape
    settings = example["diff_model"].as_dicts()[0]
    settings["geometry"][0]["mesh"] = str(replacement_mesh)
    settings["geometry"][0]["surface_selection"][0]["box"] = [[-0.501, -0.501], [-0.499, 0.501]]
    settings["output"]["directory"] = str(tmp_path / "material")
    model = diff.model(settings)
    metadata = model.metadata()
    assert (metadata.n_elements, metadata.n_dofs) == (4, 10)
    assert metadata.n_elements != example["metadata"].n_elements
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
