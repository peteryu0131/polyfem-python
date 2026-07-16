"""Small runtime helpers shared by generated examples."""

from __future__ import annotations

import time
from pathlib import Path

__all__ = ["make_timestamped_workspace"]


def make_timestamped_workspace(base_dir: Path | str, tag: str) -> Path:
    """Create ``<base_dir>/<tag>_<unix_ts>`` and return the resolved path."""
    workspace_root = Path(base_dir).resolve()
    workspace = workspace_root / f"{tag}_{int(time.time())}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace.resolve()
