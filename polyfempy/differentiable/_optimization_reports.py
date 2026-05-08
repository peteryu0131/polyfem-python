"""Small file-report writer shared by shape and material optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union


class OptimizationReportWriter:
    """Write step summaries and optional history summaries for optimization loops."""

    def __init__(
        self,
        *,
        summary_path: Optional[Union[str, Path]] = None,
        history_summary_path: Optional[Union[str, Path]] = None,
        history_formatter: Optional[Callable[[list[Any]], str]] = None,
    ) -> None:
        self.summary_path = None if summary_path is None else Path(summary_path)
        self.history_summary_path = (
            None if history_summary_path is None else Path(history_summary_path)
        )
        self.history_formatter = history_formatter
        self.report_lines: list[str] = []

    def append_step(self, step_text: str) -> None:
        self.report_lines.append(str(step_text))

    def write(self, steps: list[Any]) -> None:
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            text = "\n\n".join(self.report_lines)
            self.summary_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

        if self.history_summary_path is not None:
            if self.history_formatter is None:
                raise ValueError("history_summary_path requires a history_formatter")
            self.history_summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_summary_path.write_text(
                self.history_formatter(steps),
                encoding="utf-8",
            )


__all__ = ["OptimizationReportWriter"]
