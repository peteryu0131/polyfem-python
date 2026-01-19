"""Backend SPI for PolyFEM solver.

Backends must implement solve_impl(V, C, settings, callbacks).
Inputs: V (N, dim) float64 C-contiguous, C (M, k) int32 C-contiguous,
        settings dict, callbacks dict or None.
Returns: dict with keys {u, strain, stress, meta}.
Meta must include: backend, iters, residual, seed.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np


def solve_impl(V: np.ndarray, C: np.ndarray, settings: Dict[str, Any], 
                callbacks: Optional[Dict[str, Callable]]) -> Dict[str, Any]:
    """Backend solve implementation (documentation only).

    Backends must implement this signature.
    """
    raise NotImplementedError(
        "Backends must implement solve_impl() with this exact signature."
    )

