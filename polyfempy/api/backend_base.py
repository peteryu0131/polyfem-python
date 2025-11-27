"""Backend SPI (Service Provider Interface) for PolyFEM solver.

This module defines the interface contract that all backend implementations
(Dummy and nanobind) must follow. Backends should import this and implement
the solve_impl function with matching signature.

Interface Contract:
==================

Inputs:
    V: Vertex coordinates, shape (N, dim), dtype float64, C-contiguous.
    C: Cell connectivity, shape (M, k), dtype int32, C-contiguous.
    settings: Pure Python dict from SimulationConfig.to_dict().
    callbacks: Dict[str, callable] or None. Missing keys = no callback.

By the time inputs reach solve_impl(), they are guaranteed to satisfy:
    - V and C are numpy.ndarray
    - V.dtype == float64, V.flags.c_contiguous == True
    - C.dtype == int32, C.flags.c_contiguous == True
    - V.shape == (N, dim), C.shape == (M, k) where N, M, dim, k are positive
    - settings is a plain dict (no custom objects)
    - callbacks is a dict or None

Outputs:
    Must return a dict with these keys:
        - "u": ndarray, shape (N, dim), dtype float64, C-contiguous
        - "strain": ndarray or None
        - "stress": ndarray or None
        - "meta": dict with required keys: backend, iters, residual, seed

Meta Schema (minimum required keys):
    - backend: str ("dummy" | "nanobind")
    - iters: int (number of iterations)
    - residual: float (final residual, must be finite)
    - seed: int or None (random seed used if applicable)

Backend Responsibilities:
    1. Validate inputs if needed (though they're pre-validated by caller)
    2. Handle callbacks if present:
       - before_solve(meta): called once before iterations
       - after_iter(iter, residual, meta): called for each iteration
       - after_solve(meta): called once after iterations
    3. Produce deterministic output if random_seed is provided
    4. Return dict with keys {u, strain, stress, meta}

Example Backend Implementation:
    def solve_impl(V, C, settings, callbacks):
        # 1. Parse settings
        max_iters = settings.get("max_iters", 10)
        random_seed = settings.get("random_seed", 42)
        
        # 2. Call callbacks if present
        if callbacks and "before_solve" in callbacks:
            callbacks["before_solve"](meta={})
        
        # 3. Run solver iterations
        N, dim = V.shape
        rng = np.random.RandomState(random_seed)
        for i in range(max_iters):
            residual = 1e-3 / (i + 1)
            if callbacks and "after_iter" in callbacks:
                callbacks["after_iter"](i, residual, meta={})
        
        # 4. Generate output
        u = rng.normal(size=(N, dim)).astype(np.float64)
        
        # 5. Call final callback
        if callbacks and "after_solve" in callbacks:
            callbacks["after_solve"](meta={})
        
        # 6. Return result
        return {
            "u": u,
            "strain": None,
            "stress": None,
            "meta": {
                "backend": "dummy",
                "iters": max_iters,
                "residual": 1e-3 / max_iters,
                "seed": random_seed,
            }
        }
"""

from typing import Dict, Any, Optional, Callable
import numpy as np


def solve_impl(V: np.ndarray, C: np.ndarray, settings: Dict[str, Any], 
                callbacks: Optional[Dict[str, Callable]]) -> Dict[str, Any]:
    """Solve PolyFEM problem according to the backend SPI.
    
    This is a documentation-only function signature. Backends should implement
    this exact signature to ensure compatibility.
    
    Input Contract:
        V (vertices): np.ndarray, shape (N, dim), dtype float64, C-contiguous
        C (cells): np.ndarray, shape (M, k), dtype int32, C-contiguous
        settings: dict from SimulationConfig.to_dict() with keys:
            - pde, discr_order, stiffness, max_iters, random_seed
            - materials, bc, time, extras
        callbacks: dict mapping string to callable, or None
        
    Returns:
        dict with required keys:
            - u: np.ndarray, shape (N, dim), dtype float64, C-contiguous
            - strain: np.ndarray or None
            - stress: np.ndarray or None
            - meta: dict with required keys (backend, iters, residual, seed)
    
    Raises:
        ValueError: If inputs violate contract (though pre-validation should prevent this)
        TypeError: If callbacks return wrong types
        RuntimeError: If backend internal failure occurs
    
    Notes:
        - All inputs are pre-validated by the caller
        - Backends should trust the input contract
        - Random seed from settings should be used for determinism
        - Callbacks must be called in order: before_solve → after_iter×K → after_solve
    """
    # This function is not implemented here.
    # Backends (backend_dummy.py, backend_nanobind.py) should implement it.
    raise NotImplementedError(
        "This is a documentation-only signature. "
        "Backends must implement solve_impl() with this exact signature."
    )

