# Apply Windows-only runtime tweaks (UTF-8 console, OpenMP duplicate-lib
# toleration). The heavy lifting lives in ``_runtime.py`` so callers can invoke
# it explicitly via ``polyfempy.api.configure_windows_runtime()``. The
# auto-call here is a backward-compatibility shim only; set
# ``POLYFEMPY_SKIP_WINDOWS_AUTOCONFIG=1`` in the environment before importing
# to opt out entirely (useful for library embedders and CI).
from ._runtime import configure_windows_runtime, should_auto_configure_windows

if should_auto_configure_windows():
    configure_windows_runtime()

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
    Output, ParaviewOutput, ResultOutput, FallbackOutput,
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
    # Runtime helpers
    "configure_windows_runtime",
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
    "Output", "ParaviewOutput", "ResultOutput", "FallbackOutput",
    # Contact classes
    "Contact",
    # Problem parameter classes
    "GravityParams", "TorsionParams", "FlowParams", "FlowWithObstacleParams",
]
 
