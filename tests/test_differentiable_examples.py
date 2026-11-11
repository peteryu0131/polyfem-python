"""Check current examples, not the removed legacy example directories."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    ROOT / "differentiable_example" / "samples",
    ROOT / "differentiable_example" / "optimizations",
    ROOT / "tools" / "validation",
)
EXAMPLES = sorted(path for folder in SOURCE_DIRS for path in folder.rglob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_current_example_compiles(path):
    compile(ast.parse(path.read_text(encoding="utf-8")), str(path), "exec")


@pytest.mark.parametrize("kind", ["material", "shape"])
def test_diffdata_original_mesh_gradient_and_three_updates(kind, tmp_path):
    from tools.validation.validate_diffdata import validate

    report = validate(kind, tmp_path / kind)
    assert report["updates"] == 3
    assert report["final"] < report["initial"]
