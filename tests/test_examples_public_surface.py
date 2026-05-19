from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _example_source_files() -> list[Path]:
    return sorted(path for path in EXAMPLES.glob("*.py") if path.is_file())


def test_examples_do_not_reintroduce_removed_impact_template_name():
    offenders = []
    for path in _example_source_files():
        text = path.read_text(encoding="utf-8")
        if "make_impact_template" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
