from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_doc(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_solve_contract_doc_is_linked_from_reviewer_docs():
    assert (ROOT / "docs" / "SOLVE_CONTRACT.md").exists()

    docs_index = _read_doc("docs/README.md")
    reviewer_quickstart = _read_doc("docs/REVIEWER_QUICKSTART.md")
    test_matrix = _read_doc("docs/TEST_MATRIX.md")

    assert "SOLVE_CONTRACT.md" in docs_index
    assert "SOLVE_CONTRACT.md" in reviewer_quickstart
    assert "test_differentiable_solve_settings.py" in reviewer_quickstart
    assert "SOLVE_CONTRACT.md" in test_matrix
