from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_doc_folder_is_not_tracked_release_content():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "/doc/" in gitignore


def test_readme_documents_current_differentiable_api_boundary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Current differentiable API status" in readme
    assert "diff.shape_opt(...)" in readme
    assert "diff.model([forward_config])" in readme
    assert "diff.solve(state=..., parameters=...)" in readme
    assert "not the completed backend-backed entry point yet" in readme
