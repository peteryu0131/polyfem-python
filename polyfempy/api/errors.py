"""Error model: unified exception raising for PolyFEM API.

This module provides consistent error messages with clear prefixes
for different error categories.
"""


def raise_input_error(msg: str) -> None:
    """Raise ValueError for invalid input data.
    
    Args:
        msg: Error message describing the input violation.
        
    Raises:
        ValueError: Always raised with "INPUT: " prefix.
        
    Example:
        >>> raise_input_error("vertices must be float64 C-contiguous")
        ValueError: INPUT: vertices must be float64 C-contiguous
    """
    raise ValueError("INPUT: " + msg)


def raise_callback_type_error(msg: str) -> None:
    """Raise TypeError for invalid callback return values.
    
    Args:
        msg: Error message describing the callback violation.
        
    Raises:
        TypeError: Always raised with "CALLBACK: " prefix.
        
    Example:
        >>> raise_callback_type_error("body_force must return ndarray")
        TypeError: CALLBACK: body_force must return ndarray
    """
    raise TypeError("CALLBACK: " + msg)


def raise_backend_error(msg: str) -> None:
    """Raise RuntimeError for backend/internal failures.
    
    Args:
        msg: Error message describing the backend failure.
        
    Raises:
        RuntimeError: Always raised with "BACKEND: " prefix.
        
    Example:
        >>> raise_backend_error("solver failed to converge")
        RuntimeError: BACKEND: solver failed to converge
    """
    raise RuntimeError("BACKEND: " + msg)

