from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIFF_EXAMPLES = ROOT / "differentiable_example"


def test_differentiable_example_directory_contains_expected_files():
    files = sorted(
        path.name
        for path in DIFF_EXAMPLES.iterdir()
        if path.is_file()
    )

    assert files == ["ideal_api.md", "shapeopt_laplacian_smoke.py"]


def test_shapeopt_laplacian_smoke_example_is_output_safe():
    text = (DIFF_EXAMPLES / "shapeopt_laplacian_smoke.py").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "TemporaryDirectory" in text
    assert "differentiable_example/runs/" in gitignore
    assert '"json": ""' in text
    assert '"paraview": {"file_name": ""}' in text
    assert '"save_time_sequence": False' in text
    assert "--keep-output" in text
    assert "loss.backward()" in text
    assert "gradient_norm" in text


def test_ideal_differentiable_api_matches_forward_example_style():
    text = (DIFF_EXAMPLES / "ideal_api.md").read_text(encoding="utf-8")

    assert "SOURCE_JSON" in text
    assert "model = polyfem.model()" in text
    assert "body = model.mesh(" in text
    assert "body.material(" in text
    assert "solver = polyfem.solver(" in text
    assert "output = polyfem.output(" in text
    assert "model.config(" in text
    assert "diff_model = diff.model([model])" in text
    assert "diff_model = diff.model([polyfem_config])" not in text
    assert "objective = diff.StressNorm(selection=body, power=8)" in text
    assert "MaxStress is the intuitive meeting example" in text
    assert "objective=objective" in text
    assert "tensor=vertices" in text
    assert "def shape_loss" in text
    assert "def main()" in text
    assert "config_for_workspace" not in text
    assert "TARGET_U_PATH" not in text
    assert "polyfem_config" not in text
    assert "Optional PyTorch Loss Variants" in text
    assert "torch.linalg.norm(sol[:, -1]) * torch.linalg.norm(sol[:, -1])" in text
    assert "torch.nn.functional.mse_loss(sol, target_u)" in text
    assert "target_u is optional" in text
    assert "This does not run PolyFEM yet" in text
    assert "ShapeOpt is the torch.autograd.Function boundary" in text
    assert "loss.backward() triggers ShapeOpt.backward" in text
    assert "Objective-aware ShapeOpt" in text
    assert "the first real stress example should pass an objective into ShapeOpt" in text


def test_ideal_differentiable_api_documents_spec_alignment():
    text = (DIFF_EXAMPLES / "ideal_api.md").read_text(encoding="utf-8")

    assert "JSON Spec Alignment" in text
    assert "input-spec.json" in text
    assert "objective-spec.json" in text
    assert "polyfem.model()" in text
    assert "diff.objectives" in text
    assert "stress_norm" in text
    assert "max_stress" in text
    assert "volume_selection" in text
    assert "opt-input-spec.json" not in text
