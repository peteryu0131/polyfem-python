from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from polyfempy.differentiable._material_parameters import (  # noqa: E402
    build_lame_from_youngs,
    solver_body_ids_for_assembly,
    solver_body_slot_mask,
    youngs_to_lame,
    youngs_value_to_internal,
)


class _DirectBodyIdSolver:
    def n_element_assembly_slots(self):
        return 3

    def get_body_ids_for_assembly(self):
        return [1, 2, 1]


class _Mesh:
    def get_body_ids(self):
        return [3, 4, 3]

    def n_elements(self):
        return 3


class _MeshBodyIdSolver:
    def n_element_assembly_slots(self):
        return 2

    def mesh(self):
        return _Mesh()


def test_youngs_to_lame_scalar_values():
    E = torch.tensor(100.0, dtype=torch.float64)
    lam, mu = youngs_to_lame(E, 0.25)

    assert lam.item() == pytest.approx(40.0)
    assert mu.item() == pytest.approx(40.0)


def test_build_lame_from_youngs_with_body_slot_mask():
    mask = torch.tensor([True, False, True])
    lam, mu = build_lame_from_youngs(
        100.0,
        0.25,
        slot_mask=mask,
        other_E=200.0,
        other_nu=0.25,
    )

    assert lam.tolist() == pytest.approx([40.0, 80.0, 40.0])
    assert mu.tolist() == pytest.approx([40.0, 80.0, 40.0])


def test_youngs_value_to_internal_uses_solver_units():
    assert youngs_value_to_internal(
        3.0,
        pressure_unit="Pa",
        solver_settings={"units": {"length": "m", "mass": "kg", "time": "s"}},
    ) == pytest.approx(3.0)
    assert youngs_value_to_internal(
        1.0,
        pressure_unit="MPa",
        solver_settings={"units": {"length": "cm", "mass": "g", "time": "s"}},
    ) == pytest.approx(10000000.0)


def test_solver_body_ids_for_assembly_prefers_direct_binding():
    assert solver_body_ids_for_assembly(_DirectBodyIdSolver()).tolist() == [1, 2, 1]
    assert solver_body_slot_mask(_DirectBodyIdSolver(), body_id=1).tolist() == [
        True,
        False,
        True,
    ]


def test_solver_body_ids_for_assembly_falls_back_to_mesh_body_ids():
    assert solver_body_ids_for_assembly(_MeshBodyIdSolver()).tolist() == [3, 4]
