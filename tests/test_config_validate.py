"""Unit tests for ``SimulationConfig.validate()``.

The pre-refactor validator only accepted the plain-number form of ``E`` / ``nu``
in a single-dict ``materials`` payload. That left two common input shapes
silently broken:

    - list materials: ``materials = [{"E": 20, "nu": 0.3}, ...]``
    - unit-wrapped:   ``materials = {"E": {"value": 20, "unit": "MPa"}, "nu": 0.3}``

This test file pins down the new accepted surface and locks the failure messages
so future edits cannot quietly narrow it again.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = _REPO / "polyfempy" / "api" / "config.py"

# Mirror the loading pattern used by tests/test_config_json_io.py so we can
# exercise config.py in isolation, without pulling in the C++ backend.
SPEC = importlib.util.spec_from_file_location(
    "polyfempy_api_config_for_validate_tests", CONFIG_PATH
)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CONFIG_MODULE)
Quantity = CONFIG_MODULE.Quantity
NeoHookean = CONFIG_MODULE.NeoHookean
LinearElasticity = CONFIG_MODULE.LinearElasticity
HookeLinearElasticity = CONFIG_MODULE.HookeLinearElasticity
SaintVenant = CONFIG_MODULE.SaintVenant
SimulationConfig = CONFIG_MODULE.SimulationConfig

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class DiscrOrderTests(unittest.TestCase):
    def test_positive_int_passes(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.discr_order = 2
        cfg.validate()  # should not raise

    def test_zero_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.discr_order = 0
        with self.assertRaisesRegex(ValueError, "discr_order"):
            cfg.validate()

    def test_negative_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.discr_order = -1
        with self.assertRaisesRegex(ValueError, "discr_order"):
            cfg.validate()

    def test_non_int_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.discr_order = "two"  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "discr_order"):
            cfg.validate()


class SingleDictMaterialsTests(unittest.TestCase):
    def test_plain_numeric_passes(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.validate()

    def test_unit_wrapped_E_passes(self):
        """The JSON schema accepts E as ``{"value": 20, "unit": "MPa"}``; the
        Python validator must accept it too, otherwise cfg loaded from JSON
        fails its own validate()."""
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {
            "type": "LinearElasticity",
            "E": {"value": 20, "unit": "MPa"},
            "nu": 0.3,
        }
        cfg.validate()

    def test_unit_wrapped_with_string_value_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {
            "type": "LinearElasticity",
            "E": {"value": "twenty", "unit": "MPa"},
            "nu": 0.3,
        }
        with self.assertRaisesRegex(ValueError, r"materials\['E'\]"):
            cfg.validate()

    def test_unit_wrapper_without_value_key_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {
            "type": "LinearElasticity",
            "E": {"unit": "MPa"},
            "nu": 0.3,
        }
        with self.assertRaisesRegex(ValueError, r"materials\['E'\]"):
            cfg.validate()

    def test_string_E_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {"type": "LinearElasticity", "E": "twenty", "nu": 0.3}
        with self.assertRaisesRegex(ValueError, r"materials\['E'\]"):
            cfg.validate()

    def test_list_E_raises(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {"type": "LinearElasticity", "E": [1, 2, 3], "nu": 0.3}
        with self.assertRaisesRegex(ValueError, r"materials\['E'\]"):
            cfg.validate()

    def test_bool_E_is_rejected(self):
        """``bool`` is an ``int`` subclass, but ``E=True`` is almost certainly a
        bug; the validator rejects it explicitly."""
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {"type": "LinearElasticity", "E": True, "nu": 0.3}
        with self.assertRaisesRegex(ValueError, r"materials\['E'\]"):
            cfg.validate()


class ListMaterialsTests(unittest.TestCase):
    def _cfg_with_list(self, materials_list):
        return SimulationConfig.from_json_dict(
            {
                "pde": "LinearElasticity",
                "discr_order": 1,
                "materials": materials_list,
                "boundary_conditions": {},
                "geometry": [{"mesh": "beam.msh"}],
            }
        )

    def test_list_of_numeric_materials_passes(self):
        cfg = self._cfg_with_list(
            [
                {"type": "LinearElasticity", "E": 20, "nu": 0.3, "id": 1},
                {"type": "LinearElasticity", "E": 30, "nu": 0.4, "id": 2},
            ]
        )
        cfg.validate()

    def test_list_with_unit_wrapped_E_passes(self):
        cfg = self._cfg_with_list(
            [
                {
                    "type": "LinearElasticity",
                    "E": {"value": 20, "unit": "MPa"},
                    "nu": 0.3,
                    "id": 1,
                },
                {"type": "LinearElasticity", "E": 30, "nu": 0.4, "id": 2},
            ]
        )
        cfg.validate()

    def test_list_error_reports_index(self):
        """A bad material in the second entry must be flagged with its index so
        the user can tell which one is broken."""
        cfg = self._cfg_with_list(
            [
                {"type": "LinearElasticity", "E": 20, "nu": 0.3, "id": 1},
                {"type": "LinearElasticity", "E": "twenty", "nu": 0.4, "id": 2},
            ]
        )
        with self.assertRaisesRegex(ValueError, r"materials\[1\]\['E'\]"):
            cfg.validate()


class EdgeCaseTests(unittest.TestCase):
    def test_empty_materials_dict_passes(self):
        """No E / nu keys present at all -> nothing to validate."""
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {}
        cfg.validate()

    def test_missing_nu_still_checks_E(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {"type": "LinearElasticity", "E": "twenty"}
        with self.assertRaisesRegex(ValueError, r"materials\['E'\]"):
            cfg.validate()

    def test_bad_nu_flagged_even_when_E_is_fine(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = {"type": "LinearElasticity", "E": 20, "nu": "bad"}
        with self.assertRaisesRegex(ValueError, r"materials\['nu'\]"):
            cfg.validate()


class MaterialClassErgonomicsTests(unittest.TestCase):
    def test_quantity_serializes_to_unit_dict(self):
        self.assertEqual(
            Quantity.value(30, "MPa").to_dict(),
            {"value": 30, "unit": "MPa"},
        )

    def test_empty_neohookean_is_constructible(self):
        material = NeoHookean()
        self.assertEqual(material.to_dict(), {"type": "NeoHookean"})

    def test_partial_linear_elasticity_is_preserved_in_to_dict(self):
        material = LinearElasticity()
        material.E = {"value": 30, "unit": "MPa"}
        self.assertEqual(
            material.to_dict(),
            {"type": "LinearElasticity", "E": {"value": 30, "unit": "MPa"}},
        )

    def test_validate_rejects_incomplete_neohookean_pair(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = NeoHookean(E={"value": 20, "unit": "MPa"})
        with self.assertRaisesRegex(ValueError, "incomplete \\(E, nu\\)"):
            cfg.validate()

    def test_validate_rejects_mixed_neohookean_modes(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = NeoHookean(E=20, nu=0.3, lambda_=5, mu=8)
        with self.assertRaisesRegex(ValueError, "mixes incompatible"):
            cfg.validate()

    def test_quantity_wrapped_material_value_passes_validate(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = NeoHookean(E=Quantity.value(20, "MPa"), nu=0.3)
        cfg.validate()

    def test_empty_tensor_materials_are_constructible(self):
        self.assertEqual(HookeLinearElasticity().to_dict(), {"type": "HookeLinearElasticity"})
        self.assertEqual(SaintVenant().to_dict(), {"type": "SaintVenant"})

    def test_validate_rejects_incomplete_hooke_pair(self):
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)
        cfg.materials = HookeLinearElasticity(E=20)
        with self.assertRaisesRegex(ValueError, "incomplete \\(E, nu\\)"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
