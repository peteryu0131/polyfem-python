"""Tests for the generated-class payload boundary.

These tests stay in pure Python. They protect the first generated-only layer:
``Root.as_dict()``-style payloads should be prepared for the backend without
constructing a second config object.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from polyfempy.new_api.generated_payload import (  # noqa: E402
    generated_payload_from_config,
    prepare_generated_backend_payload,
)


class GeneratedPayloadTests(unittest.TestCase):
    def test_generated_payload_from_config_requires_dict(self):
        class BadGeneratedConfig:
            def as_dict(self):
                return ["not", "a", "dict"]

        with self.assertRaisesRegex(TypeError, "as_dict\\(\\) must return dict"):
            generated_payload_from_config(BadGeneratedConfig())

    def test_prepare_payload_does_not_mutate_input_and_preserves_backend_fields(self):
        payload = {
            "geometry": [{"mesh": "beam.msh", "enabled": True, "unused": None}],
            "materials": [{"type": "LinearElasticity", "E": 20.0, "nu": 0.3}],
            "solver": {"new_backend_option": False},
            "output": {"directory": "", "stats": False},
        }
        snapshot = copy.deepcopy(payload)

        prepared = prepare_generated_backend_payload(payload)

        self.assertEqual(payload, snapshot)
        self.assertNotIn("unused", prepared["geometry"][0])
        self.assertFalse(prepared["solver"]["new_backend_option"])
        self.assertEqual(prepared["output"]["directory"], "")
        self.assertFalse(prepared["output"]["stats"])

    def test_prepare_payload_resolves_relative_geometry_mesh_from_root_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            mesh_file = case_dir / "beam.msh"
            mesh_file.write_text("placeholder", encoding="utf-8")

            payload = {
                "root_path": str(case_dir / "config.json"),
                "geometry": [{"mesh": "beam.msh"}],
                "materials": [{"type": "LinearElasticity", "E": 20.0, "nu": 0.3}],
            }

            prepared = prepare_generated_backend_payload(payload)

        self.assertEqual(prepared["geometry"][0]["mesh"], str(mesh_file))


if __name__ == "__main__":
    unittest.main()
