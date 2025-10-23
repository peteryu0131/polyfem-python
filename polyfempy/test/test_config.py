# test/test_config.py
# Tests for polyfempy.api.config (SimulationConfig).

import importlib
import json
import pytest

from polyfempy.api.config import SimulationConfig

# ---------------------------------------------------------------------
# Section 1: Tests that do NOT require polyfempy to be installed
# ---------------------------------------------------------------------

def test_pde_aliases_canonicalization():
    """Different PDE aliases should normalize to canonical names.

    Asserts:
        - Aliases like "poisson" or "scalar" normalize to "Poisson".
        - Aliases like "elastic", "elasticity", "linear_elasticity" normalize to "LinearElasticity".
        - Confirms canonicalization handles generic scalar/tensor cases.
    """
    for alias, expected in [
        ("poisson", "Poisson"),
        ("scalar", "Poisson"),
        ("elastic", "LinearElasticity"),
        ("elasticity", "LinearElasticity"),
        ("linear_elasticity", "LinearElasticity"),
        ("linear-elasticity", "LinearElasticity"),
        ("genericscalar", "Poisson"),
        ("generictensor", "LinearElasticity"),
    ]:
        cfg = SimulationConfig(pde=alias)
        c2 = cfg.canonicalized()
        assert c2.pde == expected


def test_material_key_aliases_canonicalization():
    """Material property aliases should map to canonical keys.

    Asserts:
        - "young" maps to "E".
        - "poisson_ratio" maps to "nu".
        - If both "young" and "E" are given, the last one wins (dict order).
    """
    cfg = SimulationConfig(
        materials={
            "young": 2100,
            "poisson_ratio": 0.3,
            "E": 1000,   # last write wins
        }
    )
    mats = cfg.canonicalized().materials
    assert "E" in mats and isinstance(mats["E"], (int, float))
    assert "nu" in mats and isinstance(mats["nu"], (int, float))


def test_json_roundtrip():
    """SimulationConfig should serialize to/from JSON consistently.

    Steps:
        - Create a config with PDE, discr_order, materials, BCs, and extras.
        - Serialize to JSON string and parse back.
        - Canonicalized versions should match across key fields.

    Asserts:
        - pde, discr_order, materials, boundary_conditions, extras all match.
    """
    cfg = SimulationConfig(
        pde="elastic",
        discr_order=1,
        materials={"young": 2100, "poisson_ratio": 0.3},
        boundary_conditions={
            "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
            "rhs": [0.0, 0.1],
        },
        extras={"solver": {"linear": {"max_iter": 200}}}
    )
    s = cfg.to_json_str()
    obj = json.loads(s)
    cfg2 = SimulationConfig.from_json_str(s)

    c1 = cfg.canonicalized()
    c2 = cfg2.canonicalized()
    assert c1.pde == c2.pde
    assert c1.discr_order == c2.discr_order
    assert c1.materials == c2.materials
    assert c1.boundary_conditions == c2.boundary_conditions
    assert c1.extras == c2.extras


def test_validate_discr_order_must_be_positive_int():
    """discr_order must be a positive integer.

    Asserts:
        - ValueError is raised if discr_order is zero, negative, or non-integer.
    """
    with pytest.raises(ValueError):
        SimulationConfig(discr_order=0).validate()
    with pytest.raises(ValueError):
        SimulationConfig(discr_order=-2).validate()
    with pytest.raises(ValueError):
        SimulationConfig(discr_order="2").validate()


def test_validate_material_numbers():
    """Material values must be numeric if present.

    Asserts:
        - ValueError if E is not a number.
        - ValueError if nu is not a number.
    """
    with pytest.raises(ValueError):
        SimulationConfig(materials={"E": "2100"}).validate()
    with pytest.raises(ValueError):
        SimulationConfig(materials={"nu": "0.3"}).validate()


# ---------------------------------------------------------------------
# Section 2: Tests that REQUIRE polyfempy (auto-skipped if missing)
# ---------------------------------------------------------------------

_HAS_PF = importlib.util.find_spec("polyfempy") is not None


@pytest.mark.skipif(not _HAS_PF, reason="polyfempy not installed")
def test_to_settings_smoke_linear_elasticity():
    """Smoke test: to_settings works for LinearElasticity."""
    cfg = SimulationConfig.linear_elasticity(E=2100, nu=0.3, order=1)
    st = cfg.to_settings()
    assert st is not None


@pytest.mark.skipif(not _HAS_PF, reason="polyfempy not installed")
def test_to_settings_smoke_poisson():
    """Smoke test: to_settings works for Poisson."""
    cfg = SimulationConfig.poisson(order=1)
    st = cfg.to_settings()
    assert st is not None


@pytest.mark.skipif(not _HAS_PF, reason="polyfempy not installed")
def test_extras_pass_through_best_effort():
    """Extras should be passed through to settings (best-effort).

    Notes:
        - Should not raise even if set_advanced_option is not implemented.
    """
    cfg = SimulationConfig(
        pde="Poisson",
        discr_order=1,
        extras={"solver": {"linear": {"max_iter": 123}}}
    )
    st = cfg.to_settings()
    assert st is not None
