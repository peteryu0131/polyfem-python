"""Nanobind backend: adapter to C++ implementation.

This module attempts to import the C++ backend (polyfem_nb) and forwards
calls to solve_cpp(). If the module is not built, it raises a clear error.
"""

from . import errors as err

# Try to import C++ backend
try:
    from polyfem_nb import solve_cpp
    _cpp_backend_available = True
except (ImportError, ModuleNotFoundError):
    _cpp_backend_available = False
    solve_cpp = None


def solve_impl(V, C, settings, callbacks):
    """Nanobind backend implementation.
    
    Forwards to the C++ backend if available, otherwise raises an error.
    
    Args:
        V: Vertex coordinates, shape (N, dim), dtype float64, C-contiguous.
        C: Cell connectivity, shape (M, k), dtype int32, C-contiguous.
        settings: Dict from SimulationConfig.to_dict().
        callbacks: Dict mapping string to callable, or None.
        
    Returns:
        dict with keys: {u, strain, stress, meta}
        
    Raises:
        NotImplementedError: If C++ backend is not available.
    """
    if not _cpp_backend_available:
        raise NotImplementedError(
            "nanobind backend not connected. "
            "Please build the C++ module (polyfem_nb) first. "
            "See docs/for_cpp_dev.md for build instructions."
        )
    
    # Forward to C++ backend
    return solve_cpp(V, C, settings, callbacks)

