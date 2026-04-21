import importlib.util
import json
from pathlib import Path
import unittest
import warnings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "polyfempy" / "api" / "config.py"
SPEC = importlib.util.spec_from_file_location("polyfempy_api_config_for_tests", CONFIG_PATH)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CONFIG_MODULE)
SimulationConfig = CONFIG_MODULE.SimulationConfig


class SimulationConfigJsonIoTests(unittest.TestCase):
    def test_full_json_round_trip_uses_explicit_api(self):
        full = {
            "pde": "LinearElasticity",
            "discr_order": 2,
            "materials": [
                {
                    "type": "LinearElasticity",
                    "E": {"value": 20, "unit": "MPa"},
                    "nu": 0.3,
                }
            ],
            "boundary_conditions": {
                "dirichlet_boundary": [
                    {"id": 1, "value": [0.0, 0.0]}
                ]
            },
            "geometry": [{"mesh": "beam.msh"}],
            "time": {"dt": 0.1, "tend": 1.0},
            "output": {"directory": "results"},
        }

        cfg = SimulationConfig.from_json_dict(full)
        round_tripped = SimulationConfig.from_full_json_str(cfg.to_full_json_str())
        full_round_trip = round_tripped.to_full_json_dict()

        self.assertEqual(full_round_trip["geometry"], full["geometry"])
        self.assertEqual(full_round_trip["time"]["dt"], full["time"]["dt"])
        self.assertEqual(full_round_trip["time"]["tend"], full["time"]["tend"])
        self.assertEqual(full_round_trip["output"]["directory"], full["output"]["directory"])
        self.assertEqual(full_round_trip["materials"][0]["E"], full["materials"][0]["E"])

    def test_minimal_export_does_not_leak_private_extras(self):
        cfg = SimulationConfig.from_json_dict(
            {
                "pde": "LinearElasticity",
                "discr_order": 1,
                "materials": [{"type": "LinearElasticity", "E": 2100, "nu": 0.3}],
                "boundary_conditions": {},
                "geometry": [{"mesh": "beam.msh"}],
            }
        )
        cfg.extras["public_flag"] = 7

        exported = json.loads(cfg.to_minimal_json_str())

        self.assertNotIn("geometry", exported)
        self.assertEqual(exported.get("extras"), {"public_flag": 7})

    def test_from_json_str_auto_warns_for_legacy_compatibility(self):
        s = SimulationConfig.linear_elasticity(2100, 0.3).to_minimal_json_str()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = SimulationConfig.from_json_str(s)

        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertEqual(cfg.to_minimal_json_dict()["materials"]["E"], 2100)
        self.assertIsNone(cfg.geometry)

    def test_to_json_str_warns_and_matches_minimal_export(self):
        """`to_json_str()` must warn (it's a deprecated alias) and produce the
        same payload as the explicit `to_minimal_json_str()`.

        The symmetric test for the input side lives in
        ``test_from_json_str_auto_warns_for_legacy_compatibility``.
        """
        cfg = SimulationConfig.linear_elasticity(2100, 0.3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = cfg.to_json_str()

        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "to_json_str() is expected to emit a DeprecationWarning",
        )
        self.assertEqual(legacy, cfg.to_minimal_json_str())

    def test_from_json_str_explicit_kind_does_not_warn(self):
        """Passing ``kind='full'`` or ``kind='minimal'`` opts the caller out of
        the legacy auto-detection warning. Only ``kind='auto'`` should warn.
        """
        full_payload = SimulationConfig.from_json_dict(
            {
                "pde": "LinearElasticity",
                "discr_order": 1,
                "materials": [{"type": "LinearElasticity", "E": 2100, "nu": 0.3}],
                "boundary_conditions": {},
                "geometry": [{"mesh": "beam.msh"}],
            }
        ).to_full_json_str()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SimulationConfig.from_json_str(full_payload, kind="full")

        self.assertFalse(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "Explicit kind='full' must not emit a DeprecationWarning",
        )

    def test_from_minimal_json_str_rejects_full_only_keys(self):
        s = json.dumps(
            {
                "pde": "LinearElasticity",
                "discr_order": 1,
                "materials": {"E": 2100, "nu": 0.3},
                "boundary_conditions": {},
                "geometry": [{"mesh": "beam.msh"}],
            }
        )

        with self.assertRaisesRegex(ValueError, "from_full_json_dict"):
            SimulationConfig.from_minimal_json_str(s)


if __name__ == "__main__":
    unittest.main()
