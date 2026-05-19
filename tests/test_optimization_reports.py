from __future__ import annotations

from polyfempy.differentiable.optimization.reports import OptimizationReportWriter


def test_optimization_report_writer_writes_summary_and_history(tmp_path):
    summary_path = tmp_path / "reports" / "summary.txt"
    history_path = tmp_path / "reports" / "history.txt"
    writer = OptimizationReportWriter(
        summary_path=summary_path,
        history_summary_path=history_path,
        history_formatter=lambda steps: f"history_count: {len(steps)}\n",
    )

    writer.append_step("iter 0\nloss: 1.0")
    writer.append_step("iter 1\nloss: 0.5")
    writer.write([object(), object()])

    assert summary_path.read_text(encoding="utf-8") == (
        "iter 0\nloss: 1.0\n\niter 1\nloss: 0.5\n"
    )
    assert history_path.read_text(encoding="utf-8") == "history_count: 2\n"


def test_optimization_report_writer_noops_without_paths():
    writer = OptimizationReportWriter()

    writer.append_step("iter 0")
    writer.write([object()])

    assert writer.report_lines == ["iter 0"]
