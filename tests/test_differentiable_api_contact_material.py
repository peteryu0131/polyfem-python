"""Tiny transient cross-body material gradients; no full lattice simulation.

Two 8-triangle rectangles, 20 BDF1 steps. Only the lower body's E is trainable;
the objective is the upper body's final reference-area integral of von Mises^2.
Fixed barrier stiffness and zero friction isolate the basic contact chain.
The independent optimization check exports only initial/final tiny meshes.
"""

import json

import numpy as np
import pytest
import torch

from polyfempy import differentiable_api as diff
from polyfempy.generated_api import generated_api as pf


def _rectangle(path, x0, y0, width, height):
    points = [(x0 + width * i / 2, y0 + height * j / 2) for j in range(3) for i in range(3)]
    faces = []
    for j in range(2):
        for i in range(2):
            a = 3 * j + i + 1
            faces.extend([(a, a + 1, a + 4), (a, a + 4, a + 3)])
    path.write_text("\n".join([
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(points)),
        *[f"{i} {x} {y} 0" for i, (x, y) in enumerate(points, 1)],
        "$EndNodes", "$Elements", str(len(faces)),
        *[f"{i} 2 0 {a} {b} {c}" for i, (a, b, c) in enumerate(faces, 1)],
        "$EndElements", "",
    ]), encoding="utf-8")


def _configuration(tmp_path, *, gap=.02, contact=True, lower_E=100.):
    lower_mesh, upper_mesh = tmp_path / "lower.msh", tmp_path / "upper.msh"
    _rectangle(lower_mesh, 0, 0, 1, .25)
    _rectangle(upper_mesh, .15, .25 + gap, .7, .2)
    model = pf.model()
    lower = model.body().mesh(mesh=str(lower_mesh))
    lower.material(type="NeoHookean", E=lower_E, nu=.25, rho=1.)
    lower.surface_box(box=[[-.001, -.001], [1.001, .001]], relative=False, id=1).dirichlet(value=[0., 0.])
    lower.velocity(value=[0., 0.])
    lower.rhs(value=[0., 0.])
    upper = model.body().mesh(mesh=str(upper_mesh))
    upper.material(type="NeoHookean", E=500., nu=.25, rho=1.)
    upper.surface_all(id=2)
    upper.velocity(value=[0., -.4])
    upper.rhs(value=[0., 0.])
    config = model.config(
        space=pf.space(discr_order=1),
        time=pf.time(pf.object1(t0=0., dt=.004, tend=.08, integrator="BDF")),
        contact=pf.contact(enabled=contact, dhat=.01, friction_coefficient=0.),
        solver=pf.solver(
            max_threads=1,
            linear=pf.linear(solver="Eigen::SimplicialLDLT"),
            nonlinear=pf.nonlinear(solver="Newton", iterations_per_strategy=1,
                                   norm_type="Euclidean", rel_grad_norm_tol=0,
                                   first_grad_norm_tol=1e-12, grad_norm_tol=1e-10,
                                   line_search=pf.line_search(method="RobustArmijo")),
            contact=pf.solver_contact(barrier_stiffness=1000.),
            advanced=pf.solver_advanced(characteristic_length=1., characteristic_force_density=1.),
        ),
        output=pf.output(directory=str(tmp_path), json="",
                         paraview=pf.output_paraview(file_name=""),
                         advanced=pf.output_advanced(save_time_sequence=False, save_solve_sequence_debug=False)),
    )
    return config, lower, upper


def _problem(tmp_path, *, gap=.02, contact=True):
    config, lower, upper = _configuration(tmp_path, gap=gap, contact=contact)
    model = diff.model(config)
    metadata = model.metadata()
    assert metadata.n_elements == 16
    # Match the backend's 2D E/nu conversion (MatParams.cpp), not the 3D one.
    fixed_upper_lame = torch.tensor([500. * .25 / (1 - .25**2), 500. / (2 * (1 + .25))],
                                    dtype=torch.float64)
    objective = diff.objectives.von_mises_squared(selection=upper, weight=1.7).final()

    def loss(E):
        # In 2D: lambda=E*nu/(1-nu^2), mu=E/(2*(1+nu)).
        lower_lame = torch.stack([E * .25 / (1 - .25**2), E / (2 * (1 + .25))])
        lame = metadata.material_by_body({lower: lower_lame, upper: fixed_upper_lame})
        return diff.material_opt(model, lame, objective=objective)()

    return loss


def test_contact_transmits_lower_E_gradient_to_upper_stress(tmp_path):
    loss = _problem(tmp_path)
    E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    value = loss(E)
    value.backward()
    gradient = float(E.grad)
    assert torch.isfinite(value) and float(value.detach()) > 1e-8
    assert np.isfinite(gradient) and abs(gradient) > 1e-9
    # With zero body force and initially uniform velocity, upper-body stress
    # and sensitivity to the lower material must be transmitted through contact.
    differences = []
    for relative_step in (1e-3, 3e-4):
        h = relative_step * float(E.detach())
        plus, minus = loss(E.detach() + h), loss(E.detach() - h)
        fd = float((plus.detach() - minus.detach()) / (2 * h))
        differences.append({"relative_E_step": relative_step, "finite_difference": fd})
        assert abs(fd) > 1e-9
        assert gradient == pytest.approx(fd, rel=1e-2, abs=1e-10), differences
    assert not list(tmp_path.rglob("*.vtu"))
    print(json.dumps({"objective": "upper final integral of von Mises squared (weight 1.7)",
                      "E_lower": 100., "loss": float(value.detach()),
                      "d_loss_d_E_lower": gradient, "checks": differences}, indent=2))


@pytest.mark.parametrize("control", ["separated", "contact-disabled"])
def test_without_interaction_lower_E_does_not_affect_upper_stress(tmp_path, control):
    # At speed .4 and T=.08 the upper block travels .032; gap .2 stays open.
    loss = _problem(tmp_path, gap=.2 if control == "separated" else .02,
                    contact=control != "contact-disabled")
    E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    value = loss(E)
    value.backward()
    assert float(value.detach()) == pytest.approx(0., abs=1e-12)
    assert float(E.grad) == pytest.approx(0., abs=1e-12)
    assert not list(tmp_path.rglob("*.vtu"))


def _independent_forward_objective(output_dir, lower_E):
    """Ordinary solve with JSON E; no material_opt or adjoint evaluation."""
    from polyfempy import solve
    import meshio

    output_dir.mkdir()
    config, _, _ = _configuration(output_dir, lower_E=lower_E)
    config.output.paraview = pf.output_paraview(
        file_name="verification.pvd", skip_frame=20,
        surface=False, wireframe=False, points=False,
        options=pf.options(body_ids=True, material=True, scalar_values=True,
                           tensor_values=False, contact_forces=False),
        vismesh_rel_area=10000000,
    )
    config.output.advanced = pf.output_advanced(save_time_sequence=True, save_solve_sequence_debug=False)
    payload = config.as_dict()
    assert payload["materials"][0]["E"] == lower_E
    assert payload["materials"][1]["E"] == 500.
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    solve(cfg=config)
    mesh = meshio.read(output_dir / "step_20.vtu")
    ids = mesh.point_data["body_ids"].ravel()
    vm = mesh.point_data["von_mises"].ravel()
    assert np.isfinite(vm).all()
    np.testing.assert_allclose(mesh.point_data["E"].ravel()[ids == 2], 500.)
    np.testing.assert_allclose(mesh.point_data["E"].ravel()[ids == 1], lower_E)
    area_sum, integral = 0., 0.
    for cells in mesh.cells:
        assert cells.type == "triangle"
        labels = ids[cells.data]
        assert np.all(labels == labels[:, :1]), "A visualization triangle crosses body IDs"
        triangles = cells.data[labels[:, 0] == 2]
        points = mesh.points[triangles, :2]
        a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        area = .5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        stress = vm[triangles]
        # Affine P1 triangles with uniform material have constant Cauchy stress
        # per element. Raw (not *_avg) VTU samples retain its discontinuities.
        np.testing.assert_allclose(stress, np.repeat(stress[:, :1], 3, axis=1), rtol=1e-8, atol=1e-10)
        area_sum += float(area.sum())
        integral += float(np.dot(area, np.mean(stress**2, axis=1)))
    assert area_sum == pytest.approx(.7 * .2, rel=1e-10), "Use reference area, not deformed area"
    return 1.7 * integral  # Same objective weight as the differentiable path.


def test_one_lower_E_update_reduces_upper_objective_in_independent_forward(tmp_path):
    loss = _problem(tmp_path)
    lower_E = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    initial_E = float(lower_E.detach())
    result = (diff.optimization(lambda: loss(lower_E), parameters=[lower_E])
              .optimizer(torch.optim.Adam, lr=.1)
              .steps(1)
              .optimize(bounds=[(90., 110.)]))
    before = float(result.loss.detach())  # The shared loop records pre-update loss.
    updated_E = float(lower_E.detach())
    assert 90. <= updated_E <= 110.
    assert 0 < abs(updated_E - initial_E) <= .100001
    assert (updated_E - initial_E) * float(lower_E.grad) < 0
    after = float(loss(lower_E.detach()).detach())  # Fresh post-update solve.
    assert before - after > max(1e-8, 1e-6 * abs(before))

    forward_before = _independent_forward_objective(tmp_path / "before", initial_E)
    forward_after = _independent_forward_objective(tmp_path / "after", updated_E)
    assert forward_before == pytest.approx(before, rel=1e-6, abs=1e-9)
    assert forward_after == pytest.approx(after, rel=1e-6, abs=1e-9)
    assert forward_before - forward_after > max(1e-8, 1e-6 * abs(forward_before))
    report = {"objective": "block final reference-area integral of von Mises squared, weight 1.7",
              "lower_E_before": initial_E, "lower_E_after": updated_E,
              "fixed_block_E": 500., "d_loss_d_lower_E": float(lower_E.grad),
              "differentiable_before": before, "differentiable_after": after,
              "independent_forward_before": forward_before, "independent_forward_after": forward_after,
              "relative_reduction": (forward_before - forward_after) / forward_before}
    (tmp_path / "optimization_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
