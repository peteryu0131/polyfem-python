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
    Quantity,
    SimulationConfig,
    Material, NeoHookean, IsochoricNeoHookean, MooneyRivlin, MooneyRivlin3Param,
    MooneyRivlin3ParamSymbolic, UnconstrainedOgden, IncompressibleOgden,
    LinearElasticity, HookeLinearElasticity, SaintVenant, Stokes, NavierStokes,
    OperatorSplitting, Electrostatics, IncompressibleLinearElasticity,
    BoundaryConditions, DirichletBoundary, NeumannBoundary,
    NormalAlignedNeumannBoundary, PressureBoundary, PressureCavity,
    ObstacleDisplacement, PeriodicBoundary,
    InitialConditionEntry, InitialConditions, SoftConstraint, Constraints,
    SurfaceSelection, Body,
    Geometry, GeometryMesh, GeometryMeshArray, GeometryPlane, GeometryGround,
    GeometryMeshSequence, GeometryTransformation, GeometryAdvanced, GeometryArray,
    Solver, LinearSolver, NonlinearSolver, LineSearch, AugmentedLagrangian,
    SolverContactOptions, RayleighDamping, SolverAdvanced,
    Time, BDFIntegrator, ImplicitNewmarkIntegrator, Units,
    Output, ParaviewOutput, OutputLog, OutputParaviewOptions,
    OutputData, OutputDataAdvanced, OutputAdvanced, OutputReference,
    ResultOutput, FallbackOutput,
    Contact, CollisionMesh, Adhesion,
    Space, Tests, Input,
    GravityParams, TorsionParams, FlowParams, FlowWithObstacleParams,
)
from .result import Result
from .selection import Selection
from .batch import batch_solve
from .io import read_mesh, Mesh
from .report import (
    summarize_result,
    format_result_summary,
    summarize_history_bundle,
    format_history_bundle_txt,
    write_history_bundle_txt,
)
from .runtime import (
    make_timestamped_workspace,
    terminal_log,
    result_output,
    format_history_summary,
    write_history_artifacts,
    report_history_bundle,
    emit_history_bundle,
    solve_and_report,
)

__all__ = [
    "solve", "Quantity", "SimulationConfig", "Result", "Selection", "batch_solve",
    "read_mesh", "Mesh",
    "summarize_result", "format_result_summary",
    "summarize_history_bundle", "format_history_bundle_txt", "write_history_bundle_txt",
    "make_timestamped_workspace", "terminal_log", "result_output",
    "format_history_summary", "write_history_artifacts",
    "report_history_bundle", "emit_history_bundle", "solve_and_report",
    # Runtime helpers
    "configure_windows_runtime",
    # Material classes
    "Material", "NeoHookean", "IsochoricNeoHookean", "MooneyRivlin", "MooneyRivlin3Param",
    "MooneyRivlin3ParamSymbolic", "UnconstrainedOgden", "IncompressibleOgden",
    "LinearElasticity", "HookeLinearElasticity", "SaintVenant", "Stokes", "NavierStokes",
    "OperatorSplitting", "Electrostatics", "IncompressibleLinearElasticity",
    # Boundary condition classes
    "BoundaryConditions", "DirichletBoundary", "NeumannBoundary",
    "NormalAlignedNeumannBoundary", "PressureBoundary", "PressureCavity",
    "ObstacleDisplacement", "PeriodicBoundary",
    "InitialConditionEntry", "InitialConditions", "SoftConstraint", "Constraints",
    "SurfaceSelection", "Body",
    # Geometry classes
    "Geometry", "GeometryMesh", "GeometryMeshArray", "GeometryPlane",
    "GeometryGround", "GeometryMeshSequence", "GeometryTransformation",
    "GeometryAdvanced", "GeometryArray",
    # Solver classes
    "Solver", "LinearSolver", "NonlinearSolver", "LineSearch",
    "AugmentedLagrangian", "SolverContactOptions", "RayleighDamping",
    "SolverAdvanced",
    # Time class
    "Time", "BDFIntegrator", "ImplicitNewmarkIntegrator", "Units",
    # Output classes
    "Output", "ParaviewOutput", "OutputLog", "OutputParaviewOptions",
    "OutputData", "OutputDataAdvanced", "OutputAdvanced", "OutputReference",
    "ResultOutput", "FallbackOutput",
    # Contact classes
    "Contact", "CollisionMesh", "Adhesion",
    # Misc top-level config classes
    "Space", "Tests", "Input",
    # Problem parameter classes
    "GravityParams", "TorsionParams", "FlowParams", "FlowWithObstacleParams",
]
 
