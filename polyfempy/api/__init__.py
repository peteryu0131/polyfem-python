"""
PolyFEM Python API package.

This package provides a high-level Python interface to PolyFEM,
including configuration management and solving capabilities.
"""

from .solve import solve
from .config import SimulationConfig
from .result import Result

__all__ = ["solve", "SimulationConfig", "Result"]
