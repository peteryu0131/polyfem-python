# Fix UTF-8 encoding for Windows console (prevents Unicode math symbols from showing as garbled text)
# Fix OpenMP library conflicts (prevents libiomp5md.dll vs libomp.dll conflicts)
import sys
import os
if sys.platform == 'win32':
    try:
        import io
        # Set Python stdout/stderr to UTF-8
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        # Set console code page to UTF-8 (Windows-specific)
        os.system('chcp 65001 >nul 2>&1')
    except Exception:
        pass
    
    # Fix OpenMP library conflicts (common when PyTorch and other libraries both link OpenMP)
    # This allows the program to continue, though ideally only one OpenMP runtime should be linked
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .solve import solve
from .config import (
    SimulationConfig,
    Material, NeoHookean, IsochoricNeoHookean, MooneyRivlin, MooneyRivlin3Param,
    MooneyRivlin3ParamSymbolic, UnconstrainedOgden, IncompressibleOgden,
    LinearElasticity, HookeLinearElasticity, SaintVenant, Stokes, NavierStokes,
    OperatorSplitting, Electrostatics, IncompressibleLinearElasticity,
    BoundaryConditions, DirichletBoundary, NeumannBoundary,
    Geometry, GeometryMesh,
    Solver, LinearSolver, NonlinearSolver,
    Time,
    Output, ParaviewOutput,
    Contact,
    GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams,
)
from .result import Result
from .selection import Selection
from .batch import batch_solve
from .io import read_mesh, Mesh

__all__ = [
    "solve", "SimulationConfig", "Result", "Selection", "batch_solve",
    "read_mesh", "Mesh",
    # Material classes
    "Material", "NeoHookean", "IsochoricNeoHookean", "MooneyRivlin", "MooneyRivlin3Param",
    "MooneyRivlin3ParamSymbolic", "UnconstrainedOgden", "IncompressibleOgden",
    "LinearElasticity", "HookeLinearElasticity", "SaintVenant", "Stokes", "NavierStokes",
    "OperatorSplitting", "Electrostatics", "IncompressibleLinearElasticity",
    # Boundary condition classes
    "BoundaryConditions", "DirichletBoundary", "NeumannBoundary",
    # Geometry classes
    "Geometry", "GeometryMesh",
    # Solver classes
    "Solver", "LinearSolver", "NonlinearSolver",
    # Time class
    "Time",
    # Output classes
    "Output", "ParaviewOutput",
    # Contact classes
    "Contact",
    # Problem parameter classes
    "GravityParams", "TorsionParams", "FlowParams", "FlowWithObstacleParams",
]
 