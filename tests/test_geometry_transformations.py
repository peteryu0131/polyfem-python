"""Unit tests for ``Geometry.to_dict()`` transformation handling.

Before T5, any non-empty ``Geometry.transformations`` was silently reduced to
``transformations[0]`` and broadcast to *every* mesh, regardless of the list's
length. That produced four documented kinds of silent wrongness:

    1. length > 1: entries 1..N were silently discarded
    2. legitimate ``GeometryMesh.transformation`` values were silently
       overwritten by ``transformations[0]``
    3. the two APIs for the same concept (per-mesh vs top-level) were both
       active with conflicting semantics
    4. mismatched lengths had no error surface at all

These tests lock in the new behavior:

    - no-op when ``transformations`` is None / empty
    - length == len(meshes) zip-assigns (with DeprecationWarning)
    - length == 1 broadcasts (with DeprecationWarning, legacy shortcut)
    - any other length raises ValueError
    - per-mesh set AND top-level set for the same index raises ValueError
    - ``GeometryMesh.transformation`` alone keeps working as always (no warn)
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = _REPO / "polyfempy" / "api" / "config.py"

SPEC = importlib.util.spec_from_file_location(
    "polyfempy_api_config_for_geometry_tests", CONFIG_PATH
)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CONFIG_MODULE)
Geometry = CONFIG_MODULE.Geometry
GeometryMesh = CONFIG_MODULE.GeometryMesh

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class GeometryMeshPerMeshTransformationTests(unittest.TestCase):
    def test_per_mesh_transformation_survives_to_dict(self):
        """``GeometryMesh.transformation`` (the recommended path) is passed
        through unchanged and emits no warning."""
        geom = Geometry(
            meshes=[
                GeometryMesh(mesh="A.msh", transformation={"scale": [2, 2, 2]}),
                GeometryMesh(mesh="B.msh", transformation={"translation": [0, 1, 0]}),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = geom.to_dict()

        self.assertEqual(result[0]["transformation"], {"scale": [2, 2, 2]})
        self.assertEqual(result[1]["transformation"], {"translation": [0, 1, 0]})
        self.assertFalse(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "per-mesh transformation path should be silent",
        )

    def test_no_transformations_no_op(self):
        geom = Geometry(meshes=[GeometryMesh(mesh="A.msh"), "B.msh"])
        result = geom.to_dict()
        self.assertNotIn("transformation", result[0])
        self.assertNotIn("transformation", result[1])


class TopLevelTransformationsLegalShapesTests(unittest.TestCase):
    def test_length_matches_meshes_zip_assigns_with_warning(self):
        geom = Geometry(
            meshes=["A.msh", "B.msh", "C.msh"],
            transformations=[
                {"translation": [0, 0, 0]},
                {"translation": [0, 1, 0]},
                {"translation": [0, 2, 0]},
            ],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = geom.to_dict()

        self.assertEqual(result[0]["transformation"], {"translation": [0, 0, 0]})
        self.assertEqual(result[1]["transformation"], {"translation": [0, 1, 0]})
        self.assertEqual(result[2]["transformation"], {"translation": [0, 2, 0]})
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "zip-assign must emit a DeprecationWarning",
        )

    def test_length_one_broadcasts_with_warning(self):
        """Legacy length-1 broadcast is preserved so old scripts keep running,
        but they get a DeprecationWarning pointing at the right API."""
        geom = Geometry(
            meshes=["A.msh", "B.msh", "C.msh"],
            transformations=[{"translation": [0, 1, 0]}],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = geom.to_dict()

        for entry in result:
            self.assertEqual(entry["transformation"], {"translation": [0, 1, 0]})
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "broadcast from a single-entry list must warn",
        )

    def test_length_one_with_single_mesh_does_not_broadcast_warn(self):
        """With a single mesh, a single-entry list is not actually a broadcast,
        so we take the length-matched branch and emit the zip warning (not the
        broadcast warning). Either way the DeprecationWarning fires; the key
        invariant tested is that the value lands correctly."""
        geom = Geometry(
            meshes=["only.msh"],
            transformations=[{"scale": [3, 3, 3]}],
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = geom.to_dict()
        self.assertEqual(result[0]["transformation"], {"scale": [3, 3, 3]})


class TopLevelTransformationsErrorsTests(unittest.TestCase):
    def test_mismatched_length_raises(self):
        """The silent-drop bug's most dangerous form: 3 transformations, 5
        meshes. The old code used transformations[0] for every mesh; the new
        code raises ValueError so users see the misconfiguration."""
        geom = Geometry(
            meshes=["A.msh", "B.msh", "C.msh", "D.msh", "E.msh"],
            transformations=[
                {"translation": [0, 0, 0]},
                {"translation": [0, 1, 0]},
                {"translation": [0, 2, 0]},
            ],
        )
        with self.assertRaisesRegex(ValueError, r"length"):
            geom.to_dict()

    def test_collision_between_per_mesh_and_top_level_raises(self):
        """If a mesh already carries a GeometryMesh.transformation and the
        top-level list also wants to assign one at that index, we refuse to
        overwrite it silently (the original silent-overwrite was scenario 2
        in the T5 writeup)."""
        geom = Geometry(
            meshes=[
                GeometryMesh(mesh="A.msh", transformation={"scale": [2, 2, 2]}),
                GeometryMesh(mesh="B.msh"),
            ],
            transformations=[
                {"translation": [9, 9, 9]},
                {"translation": [0, 1, 0]},
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaisesRegex(ValueError, r"already has a transformation"):
                geom.to_dict()


class DictInputFallbackTests(unittest.TestCase):
    def test_plain_dict_mesh_entry_copies_before_mutation(self):
        """When ``meshes`` items are already plain dicts (a supported shape
        for backward compatibility), to_dict() must not mutate the caller's
        dict when applying a top-level transformation."""
        mesh_dict = {"mesh": "A.msh"}
        geom = Geometry(
            meshes=[mesh_dict],
            transformations=[{"translation": [0, 1, 0]}],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = geom.to_dict()

        self.assertEqual(result[0]["transformation"], {"translation": [0, 1, 0]})
        self.assertNotIn(
            "transformation",
            mesh_dict,
            "Geometry.to_dict() must not mutate the user-supplied dict entry",
        )


if __name__ == "__main__":
    unittest.main()
