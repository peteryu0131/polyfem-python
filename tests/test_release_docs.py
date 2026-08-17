from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_candidate_checklist_documents_submodule_and_validation_flow():
    checklist = (ROOT / "doc" / "release-candidate-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "git submodule status --recursive" in checklist
    assert "git submodule foreach --recursive git status --short" in checklist
    assert "python-from-jse" in checklist
    assert "examples" in checklist
    assert "git push origin jingyao" in checklist
    assert "python tools\\run_generated_contact_backend_checks.py" in checklist
    assert "tools/generated_contact_expected_failures.json" in checklist
    assert "PASS: 66" in checklist
    assert "IGNORED: 1" in checklist
    assert "FAIL: 0" in checklist
    assert "git clone --recurse-submodules" in checklist
    assert "python -m build" in checklist
