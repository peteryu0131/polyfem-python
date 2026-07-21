"""Optional runtime tweaks used at ``polyfempy.runtime`` import time.

Historically the package ``__init__`` silently rewrapped ``sys.stdout`` /
``sys.stderr``, ran ``chcp 65001`` and set ``KMP_DUPLICATE_LIB_OK=TRUE`` on
Windows. Useful behavior for interactive users, but bad library hygiene: a
mere ``import polyfempy`` was mutating the host process's environment.

This module isolates that behavior behind:

    - a named, idempotent, explicitly-callable function
      ``configure_windows_runtime()`` — so advanced callers can skip it, call
      it later, or audit what it touched; and
    - an environment-variable opt-out
      (``POLYFEMPY_SKIP_WINDOWS_AUTOCONFIG=1``) — so CI / library embedders
      can prevent the auto-call at import without editing code.

On non-Windows platforms the function is a no-op, so none of this affects
Linux / macOS callers (the previous behavior also guarded on ``sys.platform``).
"""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, List

__all__ = ["configure_windows_runtime", "should_auto_configure_windows"]


_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def should_auto_configure_windows() -> bool:
    """Return True iff the package should auto-apply Windows runtime tweaks
    during ``polyfempy.runtime`` import.

    Returns False when ``POLYFEMPY_SKIP_WINDOWS_AUTOCONFIG`` is set to a
    truthy value (``1``, ``true``, ``yes``, ``on``, ``y``, ``t``; any casing).
    """
    raw = os.environ.get("POLYFEMPY_SKIP_WINDOWS_AUTOCONFIG", "").strip().lower()
    return raw not in _TRUTHY


def _wrap_stream_as_utf8(stream) -> bool:
    """Rewrap a binary-buffered text stream with UTF-8. Idempotent: returns
    False if the stream is already a UTF-8 ``TextIOWrapper``.
    """
    if stream is None:
        return False
    if isinstance(stream, io.TextIOWrapper):
        try:
            if (stream.encoding or "").lower().replace("-", "") == "utf8":
                return False
        except Exception:
            pass
    buf = getattr(stream, "buffer", None)
    if buf is None:
        return False
    try:
        return io.TextIOWrapper(buf, encoding="utf-8", errors="replace") is not None
    except Exception:
        return False


def configure_windows_runtime(*, force: bool = False) -> Dict[str, List[str]]:
    """Apply Windows-only runtime tweaks to the current Python process.

    The tweaks are:

    - Rewrap ``sys.stdout`` / ``sys.stderr`` as UTF-8 ``TextIOWrapper`` so
      Unicode math symbols do not become mojibake under the default CP-1252
      console. Skipped if the stream is already UTF-8.
    - Run ``chcp 65001`` so the Windows console code page is UTF-8 too.
    - Set ``KMP_DUPLICATE_LIB_OK=TRUE`` to let a process that links multiple
      OpenMP runtimes (e.g. PyTorch + polyfempy) continue instead of aborting.
      Only set when the variable is not already in the environment so we
      never override a deliberate user choice.

    Safe to call multiple times; operations are idempotent unless
    ``force=True`` is passed (which re-wraps ``sys.stdout`` / ``sys.stderr``
    even if they are already UTF-8).

    On non-Windows platforms this is a no-op.

    Returns:
        A dict with two keys:

        - ``applied``: list of action names that this call actually performed
          (e.g. ``["stdout_utf8", "chcp_65001", "kmp_duplicate_lib_ok"]``).
        - ``skipped``: list of reasons for every action we chose not to
          perform (e.g. ``["not_windows"]``, ``["kmp_already_set"]``, or
          ``["stdout_already_utf8"]``).
    """
    applied: List[str] = []
    skipped: List[str] = []

    if sys.platform != "win32":
        skipped.append("not_windows")
        return {"applied": applied, "skipped": skipped}

    # --- UTF-8 stdout / stderr wrapping ---
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        already_utf8 = (
            isinstance(stream, io.TextIOWrapper)
            and (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8"
        )
        if already_utf8 and not force:
            skipped.append(f"{name}_already_utf8")
            continue
        buf = getattr(stream, "buffer", None)
        if buf is None:
            skipped.append(f"{name}_no_buffer")
            continue
        try:
            setattr(sys, name, io.TextIOWrapper(buf, encoding="utf-8", errors="replace"))
            applied.append(f"{name}_utf8")
        except Exception as exc:  # pragma: no cover - defensive
            skipped.append(f"{name}_utf8_error:{type(exc).__name__}")

    # --- Windows console code page ---
    try:
        os.system("chcp 65001 >nul 2>&1")
        applied.append("chcp_65001")
    except Exception as exc:  # pragma: no cover - defensive
        skipped.append(f"chcp_error:{type(exc).__name__}")

    # --- OpenMP duplicate-runtime toleration ---
    if "KMP_DUPLICATE_LIB_OK" in os.environ and not force:
        skipped.append("kmp_already_set")
    else:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        applied.append("kmp_duplicate_lib_ok")

    return {"applied": applied, "skipped": skipped}
