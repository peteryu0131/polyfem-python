from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIFF_EXAMPLES = ROOT / "differentiable_example"


def test_differentiable_example_directory_contains_only_ideal_api_doc():
    files = sorted(
        path.name
        for path in DIFF_EXAMPLES.iterdir()
        if path.is_file()
    )

    assert files == ["ideal_api.md"]


def test_ideal_differentiable_api_matches_forward_example_style():
    text = (DIFF_EXAMPLES / "ideal_api.md").read_text(encoding="utf-8")

    assert "SOURCE_JSON" in text
    assert "model = polyfem.model()" in text
    assert "body = model.mesh(" in text
    assert "body.material(" in text
    assert "solver = polyfem.solver(" in text
    assert "output = polyfem.output(" in text
    assert "polyfem_config = model.config(" in text
    assert "diff_model = diff.model([polyfem_config])" in text
    assert "def shape_loss" in text
    assert "def main()" in text
    assert "config_for_workspace" not in text
    assert "This does not run PolyFEM yet" in text
    assert "ShapeOpt is the torch.autograd.Function boundary" in text
    assert "loss.backward() triggers ShapeOpt.backward" in text
