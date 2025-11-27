"""Dummy backend: strict validation, deterministic output, callback testing.

This backend implements the solve_impl SPI contract with:
- Strict input validation (dtype, shape, contiguity)
- Deterministic pseudo-random output
- Proper callback lifecycle (before_solve → after_iter×K → after_solve)
- Error messages via unified error model
"""

import numpy as np
from typing import Dict, Any, Optional, Callable

from . import errors as err


def solve_impl(V: np.ndarray, C: np.ndarray, settings: Dict[str, Any], 
               callbacks: Optional[Dict[str, Callable]]) -> Dict[str, Any]:
    """Dummy backend implementation of the solve_impl SPI.
    
    This implementation strictly validates inputs, produces deterministic output,
    and properly triggers callbacks in the correct order.
    
    Args:
        V: Vertex coordinates, shape (N, dim), dtype float64, C-contiguous.
        C: Cell connectivity, shape (M, k), dtype int32, C-contiguous.
        settings: Dict from SimulationConfig.to_dict().
        callbacks: Dict mapping string to callable, or None.
        
    Returns:
        dict with keys: {u, strain, stress, meta}
        
    Raises:
        ValueError: If inputs violate contract (dtype/shape/contiguity).
        TypeError: If callbacks return wrong types.
        RuntimeError: If internal failure occurs (should not happen).
    """
    # ==================== INPUT VALIDATION ====================
    
    # Validate V
    if not isinstance(V, np.ndarray):
        err.raise_input_error("V must be numpy.ndarray")
    if V.dtype != np.float64:
        err.raise_input_error(f"V must be dtype float64, got {V.dtype}")
    if V.ndim != 2:
        err.raise_input_error(f"V must be 2D array, got {V.ndim}D")
    if not V.flags.c_contiguous:
        err.raise_input_error("V must be C-contiguous")
    if V.shape[0] < 1:
        err.raise_input_error(f"V first dimension must be >= 1, got {V.shape[0]}")
    if V.shape[1] not in {2, 3}:
        err.raise_input_error(f"V second dimension (dim) must be 2 or 3, got {V.shape[1]}")
    
    N, dim = V.shape
    
    # Validate C
    if not isinstance(C, np.ndarray):
        err.raise_input_error("C must be numpy.ndarray")
    if C.dtype != np.int32:
        err.raise_input_error(f"C must be dtype int32, got {C.dtype}")
    if C.ndim != 2:
        err.raise_input_error(f"C must be 2D array, got {C.ndim}D")
    if not C.flags.c_contiguous:
        err.raise_input_error("C must be C-contiguous")
    if C.shape[0] < 1:
        err.raise_input_error(f"C first dimension must be >= 1, got {C.shape[0]}")
    
    M, k = C.shape
    
    # Validate k matches dim
    if dim == 2:
        if k not in {3, 4}:
            err.raise_input_error(f"For dim=2, k must be 3 or 4, got {k}")
    else:  # dim == 3
        if k not in {4, 8}:
            err.raise_input_error(f"For dim=3, k must be 4 or 8, got {k}")
    
    # ==================== PARSE SETTINGS ====================
    max_iters = settings.get("max_iters", 10)
    random_seed = settings.get("random_seed", 42)
    
    # ==================== CALLBACK: before_solve ====================
    if callbacks and "before_solve" in callbacks:
        try:
            callbacks["before_solve"](meta={})
        except Exception as e:
            # Propagate callback exceptions
            raise
    
    # ==================== ITERATION LOOP ====================
    rng = np.random.RandomState(random_seed)
    
    for i in range(max_iters):
        # Linear decreasing residual
        residual = 1e-3 / (i + 1)
        
        # CALLBACK: after_iter
        if callbacks and "after_iter" in callbacks:
            try:
                callbacks["after_iter"](i, residual, meta={})
            except Exception as e:
                # Propagate callback exceptions
                raise
    
    # ==================== GENERATE DETERMINISTIC OUTPUT ====================
    # Generate deterministic pseudo-random displacement field
    u = rng.normal(loc=0.0, scale=1e-3, size=(N, dim)).astype(np.float64)
    
    # Ensure C-contiguous
    u = np.ascontiguousarray(u)
    
    # Final residual
    final_residual = 1e-3 / max_iters
    
    # ==================== CALLBACK: after_solve ====================
    if callbacks and "after_solve" in callbacks:
        try:
            callbacks["after_solve"](meta={})
        except Exception as e:
            # Propagate callback exceptions
            raise
    
    # ==================== RETURN RESULT ====================
    return {
        "u": u,
        "strain": None,
        "stress": None,
        "meta": {
            "backend": "dummy",
            "iters": max_iters,
            "residual": final_residual,
            "seed": random_seed,
        }
    }

