"""Public Python API for PolyFEM.

The recommended user-facing surface is intentionally small:

- ``solve`` for running simulations
- ``SimulationConfig`` for structured configuration
- ``Result`` for structured solver output

The additional config classes and reporting/runtime helpers remain importable
for backward compatibility and for advanced users who need lower-level control.
Documentation should treat them as secondary APIs, not as the main entry path.
"""

# Apply Windows-only runtime tweaks (UTF-8 console, OpenMP duplicate-lib
# toleration). The heavy lifting lives in ``_runtime.py`` so callers can invoke
# it explicitly via ``polyfempy.api.configure_windows_runtime()``. The auto-call
# here is a backward-compatibility shim only; set
# ``POLYFEMPY_SKIP_WINDOWS_AUTOCONFIG=1`` before importing to opt out.
from ._runtime import configure_windows_runtime, should_auto_configure_windows

if should_auto_configure_windows():
    configure_windows_runtime()

from .config import (
    Adhesion,
    AugmentedLagrangian,
    BDFIntegrator,
    Body,
    BoundaryConditions,
    CollisionMesh,
    Constraints,
    Contact,
    DirichletBoundary,
    Electrostatics,
    FallbackOutput,
    FlowParams,
    FlowWithObstacleParams,
    Geometry,
    GeometryAdvanced,
    GeometryArray,
    GeometryGround,
    GeometryMesh,
    GeometryMeshArray,
    GeometryMeshSequence,
    GeometryPlane,
    GeometryTransformation,
    GravityParams,
    HookeLinearElasticity,
    ImplicitNewmarkIntegrator,
    IncompressibleLinearElasticity,
    IncompressibleOgden,
    InitialConditionEntry,
    InitialConditions,
    Input,
    IsochoricNeoHookean,
    LineSearch,
    LinearElasticity,
    LinearSolver,
    Material,
    MooneyRivlin,
    MooneyRivlin3Param,
    MooneyRivlin3ParamSymbolic,
    NavierStokes,
    NeoHookean,
    NeumannBoundary,
    NonlinearSolver,
    NormalAlignedNeumannBoundary,
    ObstacleDisplacement,
    OperatorSplitting,
    Output,
    OutputAdvanced,
    OutputData,
    OutputDataAdvanced,
    OutputLog,
    OutputParaviewOptions,
    OutputReference,
    ParaviewOutput,
    PeriodicBoundary,
    PressureBoundary,
    PressureCavity,
    Quantity,
    RayleighDamping,
    ResultOutput,
    SaintVenant,
    SimulationConfig,
    SoftConstraint,
    Solver,
    SolverAdvanced,
    SolverContactOptions,
    Space,
    Stokes,
    SurfaceSelection,
    Tests,
    Time,
    TorsionParams,
    UnconstrainedOgden,
    Units,
)
from .io import Mesh, read_mesh
from .report import (
    format_result_summary,
    format_history_bundle_txt,
    summarize_history_bundle,
    summarize_result,
    write_history_bundle_txt,
)
from .result import Result
from .runtime import (
    format_history_summary,
    emit_history_bundle,
    make_timestamped_workspace,
    report_history_bundle,
    result_output,
    solve_and_report,
    terminal_log,
    write_history_artifacts,
)
from .selection import Selection
from .solve import solve

CORE_API = [
    "solve",
    "SimulationConfig",
    "Result",
]

# These groups are still exported so older scripts keep working. They are not
# the recommended first path for new users; prefer CORE_API plus
# ``polyfempy.api.guided`` for guided configuration.
IO_API = [
    "Selection",
    "Mesh",
    "read_mesh",
]

REPORTING_API = [
    "summarize_result",
    "format_result_summary",
    "summarize_history_bundle",
    "format_history_bundle_txt",
    "write_history_bundle_txt",
]

RUNTIME_API = [
    "make_timestamped_workspace",
    "terminal_log",
    "result_output",
    "format_history_summary",
    "write_history_artifacts",
    "report_history_bundle",
    "emit_history_bundle",
    "solve_and_report",
]

CONFIG_API = [
    "Quantity",
    "Material",
    "NeoHookean",
    "IsochoricNeoHookean",
    "MooneyRivlin",
    "MooneyRivlin3Param",
    "MooneyRivlin3ParamSymbolic",
    "UnconstrainedOgden",
    "IncompressibleOgden",
    "LinearElasticity",
    "HookeLinearElasticity",
    "SaintVenant",
    "Stokes",
    "NavierStokes",
    "OperatorSplitting",
    "Electrostatics",
    "IncompressibleLinearElasticity",
    "BoundaryConditions",
    "DirichletBoundary",
    "NeumannBoundary",
    "NormalAlignedNeumannBoundary",
    "PressureBoundary",
    "PressureCavity",
    "ObstacleDisplacement",
    "PeriodicBoundary",
    "InitialConditionEntry",
    "InitialConditions",
    "SoftConstraint",
    "Constraints",
    "SurfaceSelection",
    "Body",
    "Geometry",
    "GeometryMesh",
    "GeometryMeshArray",
    "GeometryPlane",
    "GeometryGround",
    "GeometryMeshSequence",
    "GeometryTransformation",
    "GeometryAdvanced",
    "GeometryArray",
    "Solver",
    "LinearSolver",
    "NonlinearSolver",
    "LineSearch",
    "AugmentedLagrangian",
    "SolverContactOptions",
    "RayleighDamping",
    "SolverAdvanced",
    "Time",
    "BDFIntegrator",
    "ImplicitNewmarkIntegrator",
    "Units",
    "Output",
    "ParaviewOutput",
    "OutputLog",
    "OutputParaviewOptions",
    "OutputData",
    "OutputDataAdvanced",
    "OutputAdvanced",
    "OutputReference",
    "ResultOutput",
    "FallbackOutput",
    "Contact",
    "CollisionMesh",
    "Adhesion",
    "Space",
    "Tests",
    "Input",
    "GravityParams",
    "TorsionParams",
    "FlowParams",
    "FlowWithObstacleParams",
]

WINDOWS_RUNTIME_API = [
    "configure_windows_runtime",
]

ADVANCED_COMPAT_API = (
    IO_API
    + REPORTING_API
    + RUNTIME_API
    + CONFIG_API
    + WINDOWS_RUNTIME_API
)

__all__ = (
    CORE_API
    + ADVANCED_COMPAT_API
)
