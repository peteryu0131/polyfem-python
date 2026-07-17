from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _example_source_files() -> list[Path]:
    return sorted(path for path in EXAMPLES.rglob("*.py") if path.is_file())


def test_examples_do_not_reintroduce_removed_impact_template_name():
    offenders = []
    for path in _example_source_files():
        text = path.read_text(encoding="utf-8")
        if "make_impact_template" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_classic_examples_use_generated_api_helpers_only():
    expected = {
        "examples/classic_example/2D/new_better_contact_2d_golf_ball_deformable_wall_generated_api.py",
        "examples/classic_example/2D/_contact_2d_common.py",
        "examples/classic_example/3D/contact_3d_codimensional_mat_knives_generated_api.py",
        "examples/classic_example/3D/_contact_3d_common.py",
    }
    actual = {path.relative_to(ROOT).as_posix() for path in _example_source_files()}
    assert expected <= actual


def test_classic_examples_do_not_import_core_runtime_helper():
    offenders = []
    for path in _example_source_files():
        text = path.read_text(encoding="utf-8")
        if "polyfempy.api.runtime" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
