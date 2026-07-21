from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_differentiable_package_is_marked_reference_only():
    readme = ROOT / "polyfempy" / "differentiable" / "README.md"
    text = readme.read_text(encoding="utf-8").lower()

    assert "reference" in text
    assert "experimental" in text
    assert "not part of the current supported polyfempy.runtime interface" in text
    assert "simulationconfig" in text
