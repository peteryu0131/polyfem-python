"""Public guided section support and presets.

This module contains the implementation behind ``polyfempy.api.guided``.  It is
kept inside the ``polyfempy`` package so installed users do not depend on the
repository-local ``experiment`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import numpy as np

from polyfempy.api import (
    Adhesion,
    CollisionMesh,
    Contact,
    Geometry,
    GeometryAdvanced,
    GeometryGround,
    GeometryMesh,
    GeometryMeshSequence,
    GeometryPlane,
    GeometryTransformation,
    LinearElasticity,
    LinearSolver,
    NeoHookean,
    NonlinearSolver,
    Output,
    OutputAdvanced,
    OutputData,
    OutputDataAdvanced,
    OutputLog,
    OutputParaviewOptions,
    OutputReference,
    ParaviewOutput,
    SaintVenant,
    SimulationConfig,
    Solver,
    SolverContactOptions,
    Space,
    SurfaceSelection,
    Time,
    Units,
)


HERE = Path(__file__).resolve().parent
_PACKAGE_ROOT = HERE.parents[1]
_LEGACY_EXPERIMENT_MESH_DIR = (
    _PACKAGE_ROOT / "experiment" / "experiment_api_solve" / "meshes"
)
MESH_DIR = (
    _LEGACY_EXPERIMENT_MESH_DIR
    if _LEGACY_EXPERIMENT_MESH_DIR.exists()
    else HERE / "meshes"
)
DEFAULT_REQUESTED_FIELDS = ("u", "stress", "von_mises")


def mesh_file(name: str) -> str:
    return str((MESH_DIR / name).resolve())


PDEName: TypeAlias = Literal[
    "Poisson",
    "LinearElasticity",
    "NonLinearElasticity",
    "Helmholtz",
    "Bilaplacian",
    "Stokes",
]
ProblemTypeName: TypeAlias = Literal[
    "Gravity",
    "Franke",
    "TorsionElastic",
    "Flow",
    "DrivenCavity",
    "FlowWithObstacle",
]
LengthUnitName: TypeAlias = Literal[
    "km",
    "m",
    "dm",
    "cm",
    "mm",
    "um",
    "nm",
    "in",
    "ft",
]
MassUnitName: TypeAlias = Literal["t", "kg", "g", "mg", "lb"]
TimeUnitName: TypeAlias = Literal["h", "min", "s", "ms", "us", "ns"]
PressureUnitName: TypeAlias = Literal["Pa", "kPa", "MPa", "GPa"]
DensityUnitName: TypeAlias = Literal["kg/m^3", "g/cm^3", "kg/mm^3"]
ResultFieldName: TypeAlias = Literal[
    "u",
    "p",
    "pressure",
    "stress",
    "strain",
    "von_mises",
    "von_mises_avg",
    "body_ids",
    "velocity",
    "acceleration",
]
GeometryExtractName: TypeAlias = Literal["volume", "edges", "points", "surface"]
BasisTypeName: TypeAlias = Literal["Lagrange", "Spline", "Serendipity", "Bernstein"]
PolyBasisTypeName: TypeAlias = Literal["MFSHarmonic", "MeanValue", "Wachspress"]
BoundaryConditionMethodName: TypeAlias = Literal["lsq", "sample"]
AxisSideName: TypeAlias = Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
SurfaceSelectionModeName: TypeAlias = Literal["position", "sphere", "box", "plane"]
RotationModeName: TypeAlias = Literal["xyz", "axis_angle", "quaternion", "rotation_vector"]
MaterialModelName: TypeAlias = Literal["NeoHookean", "LinearElasticity", "SaintVenant"]
MaterialModeName: TypeAlias = Literal["young_poisson", "lame"]
YoungPoissonModelName: TypeAlias = Literal["NeoHookean", "LinearElasticity", "SaintVenant"]
LameModelName: TypeAlias = Literal["NeoHookean", "LinearElasticity"]
LogLevelName: TypeAlias = Literal["trace", "debug", "info", "warn", "warning", "error", "critical", "off"]
LinearSolverName: TypeAlias = Literal[
    "Eigen::SimplicialLDLT",
    "Eigen::SparseLU",
    "Eigen::CholmodSupernodalLLT",
    "Eigen::UmfPackLU",
    "Eigen::SuperLU",
    "Eigen::PardisoLDLT",
    "Eigen::PardisoLLT",
    "Eigen::PardisoLU",
    "Pardiso",
    "Hypre",
    "AMGCL",
    "Eigen::LeastSquaresConjugateGradient",
    "Eigen::DGMRES",
    "Eigen::ConjugateGradient",
    "Eigen::BiCGSTAB",
    "Eigen::GMRES",
    "Eigen::MINRES",
]
LinearPreconditionerName: TypeAlias = Literal[
    "Eigen::IdentityPreconditioner",
    "Eigen::DiagonalPreconditioner",
    "Eigen::IncompleteCholesky",
    "Eigen::LeastSquareDiagonalPreconditioner",
    "Eigen::IncompleteLUT",
]
NonlinearSolverName: TypeAlias = Literal[
    "Newton",
    "DenseNewton",
    "GradientDescent",
    "ADAM",
    "StochasticADAM",
    "StochasticGradientDescent",
    "L-BFGS",
    "BFGS",
    "L-BFGS-B",
    "MMA",
]
LineSearchName: TypeAlias = Literal["backtracking", "armijo", "wolfe", "none"]
TimeIntegratorName: TypeAlias = Literal[
    "ImplicitEuler",
    "BDF1",
    "BDF2",
    "BDF3",
    "BDF4",
    "BDF5",
    "BDF6",
    "ImplicitNewmark",
]
ContactModeName: TypeAlias = Literal["disabled", "frictionless", "coulomb", "adhesive"]
BarrierStiffnessName: TypeAlias = Literal["adaptive"]
CollisionTessellationTypeName: TypeAlias = Literal["regular", "irregular"]
CCDBroadPhaseName: TypeAlias = Literal[
    "hash_grid",
    "HG",
    "brute_force",
    "BF",
    "spatial_hash",
    "SH",
    "bvh",
    "BVH",
    "sweep_and_prune",
    "SAP",
    "sweep_and_tiniest_queue",
    "STQ",
]


@dataclass
class ProblemSection:
    """Top-level PDE choice shown via IDE autocomplete.

    Kept as its own section on purpose: even though it is currently small, this
    is the natural place where users choose what kind of problem they want to
    solve.
    """

    pde: PDEName = "NonLinearElasticity"
    problem_type: ProblemTypeName | None = None
    problem_params: dict | None = None


@dataclass
class UnitsSection:
    """Basic unit-system settings.

    The schema only specifies these as strings, so the guided API exposes a
    curated set of common choices for IDE autocomplete while still defaulting
    to the schema defaults.
    """

    length: LengthUnitName = "m"
    mass: MassUnitName = "kg"
    time: TimeUnitName = "s"
    characteristic_length: float | None = 1.0


@dataclass
class MaterialSection:
    """Material family + parameterization mode."""

    model: MaterialModelName = "NeoHookean"
    mode: MaterialModeName = "young_poisson"
    E: float = 20.0
    E_unit: PressureUnitName = "MPa"
    nu: float = 0.45
    rho: float = 1100.0
    rho_unit: DensityUnitName = "kg/m^3"
    lambda_: float | None = None
    lambda_unit: PressureUnitName = "Pa"
    mu: float | None = None
    mu_unit: PressureUnitName = "Pa"


@dataclass
class FixedSurfaceSection:
    mode: SurfaceSelectionModeName = "position"
    enabled: bool = True
    axis: int = -2
    side: AxisSideName | None = None
    position: float = 0.0001
    relative: bool = True
    center: tuple[float, ...] | None = None
    radius: float | None = None
    box_min: tuple[float, ...] | None = None
    box_max: tuple[float, ...] | None = None
    normal: tuple[float, ...] | None = None
    offset: float | None = None
    value: tuple[float, ...] = (0.0, 0.0)


@dataclass
class BodySection:
    name: str
    material: MaterialSection
    mesh: str | None = None
    vertices: Any | None = None
    cells: Any | None = None
    extract: GeometryExtractName = "volume"
    unit: str = ""
    n_refs: int = 0
    enabled: bool = True
    is_obstacle: bool = False
    transformation: "TransformationSection | None" = None
    advanced: "GeometryAdvancedSection | None" = None
    fixed_surfaces: list[FixedSurfaceSection] = field(default_factory=list)
    initial_velocity: tuple[float, float] | None = None
    initial_solution: tuple[float, float] | None = None
    initial_acceleration: tuple[float, float] | None = None


@dataclass
class LoadsSection:
    rhs: tuple[float, float] = (0.0, 0.0)


@dataclass
class SpaceSection:
    """Discretization-space settings.

    In the guided API, ``discr_order`` lives here so users only set the
    polynomial order in one place.
    """

    discr_order: int | None = 1
    pressure_discr_order: int | None = 1
    basis_type: BasisTypeName | None = "Lagrange"
    poly_basis_type: PolyBasisTypeName | None = "MFSHarmonic"
    use_p_ref: bool | None = False
    bc_method: BoundaryConditionMethodName | None = "sample"
    advanced: dict | None = None


@dataclass
class TimeSection:
    t0: float = 0.0
    tend: float = 0.02
    dt: float = 0.01
    time_steps: int | None = None
    integrator: TimeIntegratorName = "ImplicitEuler"
    bdf_steps: int = 1
    gamma: float = 0.5
    beta: float = 0.25
    quasistatic: bool = False


@dataclass
class LinearSolverSection:
    solver: LinearSolverName = "Eigen::PardisoLDLT"
    precond: LinearPreconditionerName | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass
class NonlinearSolverSection:
    solver: NonlinearSolverName = "Newton"
    tolerance: float = 1e-6
    grad_norm: float = 0.002
    x_delta: float | None = None
    max_iterations: int = 800
    iterations_per_strategy: int | None = None
    line_search: LineSearchName | None = None
    residual_tolerance: float = 100.0
    history_size: int | None = None


@dataclass
class SolverContactSection:
    barrier_stiffness: BarrierStiffnessName | float = "adaptive"
    initial_barrier_stiffness: float | None = None
    friction_iterations: int | None = None
    tangential_adhesion_iterations: int | None = None
    friction_convergence_tol: float | None = None
    ccd_broad_phase: CCDBroadPhaseName = "hash_grid"
    ccd_tolerance: float = 1e-6
    ccd_max_iterations: int = 1_000_000


@dataclass
class SolverSection:
    linear: LinearSolverSection = field(default_factory=LinearSolverSection)
    nonlinear: NonlinearSolverSection = field(default_factory=NonlinearSolverSection)
    contact: SolverContactSection = field(default_factory=SolverContactSection)


@dataclass
class PlaneObstacleSection:
    point: tuple[float, ...]
    normal: tuple[float, ...]
    enabled: bool = True


@dataclass
class GroundObstacleSection:
    height: float = 0.0
    enabled: bool = True


@dataclass
class MeshSequenceSection:
    files: list[str] = field(default_factory=list)
    fps: int = 1
    extract: GeometryExtractName = "volume"
    unit: str = ""
    n_refs: int = 0
    enabled: bool = True
    is_obstacle: bool = True


@dataclass
class TransformationSection:
    translation: tuple[float, ...] = ()
    rotation: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    dimensions: float | tuple[float, ...] = 1.0
    rotation_mode: RotationModeName = "xyz"


@dataclass
class GeometryAdvancedSection:
    normalize_mesh: bool = False
    force_linear_geometry: bool = False
    refinement_location: float = 0.5
    min_component: int = -1


@dataclass
class ResultsSection:
    """Python-side result fields requested from ``solve()`` / ``solve_differentiable()``."""

    requested_fields: list[ResultFieldName | str] = field(
        default_factory=lambda: list(DEFAULT_REQUESTED_FIELDS)
    )
    strict: bool = False


@dataclass
class OutputFilesSection:
    save_vtu: bool = False
    json_name: str = ""
    restart_json_name: str = ""


@dataclass
class OutputLogSection:
    level: LogLevelName = "debug"
    file_level: LogLevelName = "trace"
    path: str = ""
    quiet: bool = False


@dataclass
class ParaviewFieldsSection:
    use_hdf5: bool = False
    material: bool = False
    body_ids: bool = False
    contact_forces: bool = False
    friction_forces: bool = False
    normal_adhesion_forces: bool = False
    tangential_adhesion_forces: bool = False
    velocity: bool = False
    acceleration: bool = False
    scalar_values: bool = True
    tensor_values: bool = True
    discretization_order: bool = True
    nodes: bool = True
    forces: bool = False
    force_high_order: bool = False
    jacobian_validity: bool = False


@dataclass
class ParaviewSection:
    file_name: str = ""
    volume: bool = True
    surface: bool = False
    wireframe: bool = False
    points: bool = False
    vismesh_rel_area: float = 1e-5
    skip_frame: int = 1
    high_order_mesh: bool = True
    fields: list[str] = field(default_factory=list)
    options: ParaviewFieldsSection = field(default_factory=ParaviewFieldsSection)


@dataclass
class OutputDataSection:
    solution: str = ""
    full_mat: str = ""
    stiffness_mat: str = ""
    stress_mat: str = ""
    state: str = ""
    rest_mesh: str = ""
    mises: str = ""
    nodes: str = ""
    reorder_nodes: bool = False
    file_index_offset: int = 0


@dataclass
class OutputReferenceSection:
    solution: list[str] = field(default_factory=list)
    gradient: list[str] = field(default_factory=list)


@dataclass
class OutputAdvancedSection:
    timestep_prefix: str = "step_"
    sol_on_grid: float = -1.0
    compute_error: bool = True
    sol_at_node: int = -1
    vis_boundary_only: bool = False
    curved_mesh_size: bool = False
    save_solve_sequence_debug: bool = False
    save_ccd_debug_meshes: bool = False
    save_time_sequence: bool = True
    save_nl_solve_sequence: bool = False
    spectrum: bool = False


@dataclass
class CollisionMeshSection:
    enabled: bool = True
    tessellation_type: CollisionTessellationTypeName = "regular"
    mesh: str | None = None
    linear_map: str | None = None
    max_edge_length: float | None = None


@dataclass
class AdhesionSection:
    enabled: bool = False
    dhat_p: float = 0.001
    dhat_a: float = 0.01
    adhesion_strength: float = 0.001
    tangential_adhesion_coefficient: float = 0.0
    epsa: float = 0.001


@dataclass
class OutputSection:
    directory: str = ""
    stats: bool = False
    files: OutputFilesSection = field(default_factory=OutputFilesSection)
    log: OutputLogSection = field(default_factory=OutputLogSection)
    paraview: ParaviewSection = field(default_factory=ParaviewSection)
    data: OutputDataSection = field(default_factory=OutputDataSection)
    reference: OutputReferenceSection = field(default_factory=OutputReferenceSection)
    advanced: OutputAdvancedSection = field(default_factory=OutputAdvancedSection)


@dataclass
class ContactSection:
    mode: ContactModeName = "disabled"
    dhat: float = 0.001
    mu: float = 0.0
    dhat_percentage: float = 0.8
    epsv: float = 0.001
    use_convergent_formulation: bool = False
    use_area_weighting: bool = True
    use_improved_max_operator: bool = True
    use_physical_barrier: bool = True
    use_gcp_formulation: bool = False
    alpha_n: float = 0.5
    alpha_t: float = 0.5
    min_distance_ratio: float = 0.5
    use_adaptive_dhat: bool = False
    periodic: bool = False
    barrier_stiffness: str | float | None = None
    collision_mesh: CollisionMeshSection | None = None
    adhesion: AdhesionSection | None = None
    adhesion_strength: float = 0.001


@dataclass
class ExperimentTemplate:
    problem: ProblemSection = field(default_factory=ProblemSection)
    units: UnitsSection = field(default_factory=UnitsSection)
    bodies: list[BodySection] = field(default_factory=list)
    space: SpaceSection = field(default_factory=SpaceSection)
    geometry_extras: list[PlaneObstacleSection | GroundObstacleSection | MeshSequenceSection] = field(default_factory=list)
    loads: LoadsSection = field(default_factory=LoadsSection)
    time: TimeSection = field(default_factory=TimeSection)
    solver: SolverSection = field(default_factory=SolverSection)
    contact: ContactSection = field(default_factory=ContactSection)
    results: ResultsSection = field(default_factory=ResultsSection)
    output: OutputSection = field(default_factory=OutputSection)


ImpactTemplate = ExperimentTemplate


def default_impact_bodies() -> list[BodySection]:
    return [
        body_section(
            name="lattice",
            mesh="triangular_lattice.msh",
            material=material_section(
                model="NeoHookean",
                mode="young_poisson",
                E=20.0,
                E_unit="MPa",
                nu=0.45,
                rho=1100.0,
                rho_unit="kg/m^3",
            ),
            fixed_surface=fixed_surface_section(side="y_min"),
        ),
        body_section(
            name="block",
            mesh="falling_weight_block.msh",
            material=material_section(
                model="NeoHookean",
                mode="young_poisson",
                E=200.0,
                E_unit="GPa",
                nu=0.45,
                rho=7850.0,
                rho_unit="kg/m^3",
            ),
            initial_velocity=(0.0, 0.0),
        ),
    ]


def default_impact_template() -> ExperimentTemplate:
    """Create a fully-filled runnable preset for the impact example."""
    return experiment_template(
        problem=problem_section(
            pde="NonLinearElasticity",
        ),
        units=units_section(
            length="cm",
            mass="g",
            time="s",
            characteristic_length=1.0,
        ),
        bodies=default_impact_bodies(),
        space=space_section(discr_order=1),
        loads=loads_section(rhs=(0.0, 980.0)),
        time=time_section(t0=0.0, tend=0.02, dt=0.01),
        solver=solver_section(
            linear=linear_solver_section(solver="Eigen::PardisoLDLT"),
            nonlinear=nonlinear_solver_section(
                solver="Newton",
                max_iterations=800,
                grad_norm=0.002,
                residual_tolerance=100.0,
            ),
            contact=solver_contact_section(barrier_stiffness="adaptive"),
        ),
        contact=contact_section(mode="frictionless", dhat=0.012),
        results=results_section(requested_fields=["u", "stress", "von_mises"]),
        output=output_section(
            directory=".",
            files=output_files_section(
                save_vtu=True,
                json_name="impact_stats.json",
            ),
            log=output_log_section(
                level="debug",
                file_level="debug",
                path="polyfem.log",
            ),
            paraview=paraview_section(
                file_name="impact.pvd",
                vismesh_rel_area=10_000_000,
                options=paraview_fields_section(
                    material=True,
                    body_ids=True,
                    velocity=True,
                    scalar_values=True,
                    tensor_values=True,
                ),
            ),
            advanced=output_advanced_section(
                timestep_prefix="impact_step_",
                save_time_sequence=True,
            ),
        ),
    )


def problem_section(
    *,
    pde: PDEName = "NonLinearElasticity",
    problem_type: ProblemTypeName | None = None,
    problem_params: dict | None = None,
) -> ProblemSection:
    """Problem-level choices.

    In the guided API, polynomial order is configured in ``space_section(...)``
    instead of here.

    Common guided choices:
    - ``"NonLinearElasticity"``
    - ``"LinearElasticity"``
    - ``"Poisson"``
    - ``"Stokes"``

    Optional ``problem_type`` exposes predefined problems such as
    ``"Gravity"`` or ``"FlowWithObstacle"`` when you want a preset-style
    backend problem instead of only setting a PDE label.
    """
    return ProblemSection(
        pde=pde,
        problem_type=problem_type,
        problem_params=problem_params,
    )


def units_section(
    *,
    length: LengthUnitName = "m",
    mass: MassUnitName = "kg",
    time: TimeUnitName = "s",
    characteristic_length: float | None = 1.0,
) -> UnitsSection:
    """Create a units section with schema-backed defaults.

    Leave this alone for standard SI defaults. Change it only when the
    experiment is more naturally described in another unit system such as CGS.
    """
    return UnitsSection(
        length=length,
        mass=mass,
        time=time,
        characteristic_length=characteristic_length,
    )


@overload
def material_section(
    *,
    model: YoungPoissonModelName = "NeoHookean",
    mode: Literal["young_poisson"] = "young_poisson",
    E: float = 20.0,
    E_unit: PressureUnitName = "MPa",
    nu: float = 0.45,
    rho: float = 1100.0,
    rho_unit: DensityUnitName = "kg/m^3",
) -> MaterialSection:
    ...


@overload
def material_section(
    *,
    model: LameModelName = "NeoHookean",
    mode: Literal["lame"],
    lambda_: float,
    lambda_unit: PressureUnitName = "Pa",
    mu: float,
    mu_unit: PressureUnitName = "Pa",
    rho: float = 1100.0,
    rho_unit: DensityUnitName = "kg/m^3",
) -> MaterialSection:
    ...


def material_section(
    *,
    model: MaterialModelName = "NeoHookean",
    mode: MaterialModeName = "young_poisson",
    E: float = 20.0,
    E_unit: PressureUnitName = "MPa",
    nu: float = 0.45,
    rho: float = 1100.0,
    rho_unit: DensityUnitName = "kg/m^3",
    lambda_: float | None = None,
    lambda_unit: PressureUnitName = "Pa",
    mu: float | None = None,
    mu_unit: PressureUnitName = "Pa",
) -> MaterialSection:
    """Create a material section.

    Common use:
    - ``mode="young_poisson"`` with ``E`` / ``nu``
    - ``mode="lame"`` with ``lambda_`` / ``mu``

    IDE autocomplete is provided for common material families and units.
    """
    return MaterialSection(
        model=model,
        mode=mode,
        E=E,
        E_unit=E_unit,
        nu=nu,
        rho=rho,
        rho_unit=rho_unit,
        lambda_=lambda_,
        lambda_unit=lambda_unit,
        mu=mu,
        mu_unit=mu_unit,
    )


def fixed_surface_section(
    *,
    mode: SurfaceSelectionModeName = "position",
    enabled: bool = True,
    axis: int = -2,
    side: AxisSideName | None = None,
    position: float = 0.0001,
    relative: bool = True,
    center: tuple[float, ...] | None = None,
    radius: float | None = None,
    box_min: tuple[float, ...] | None = None,
    box_max: tuple[float, ...] | None = None,
    normal: tuple[float, ...] | None = None,
    offset: float | None = None,
    value: tuple[float, ...] = (0.0, 0.0),
) -> FixedSurfaceSection:
    """Create a fixed-surface Dirichlet condition selector.

    Most users will use the default ``mode="position"`` with either:
    - ``side="y_min"`` style human-readable selection, or
    - ``axis=-2`` style JSON-compatible selection.

    Other modes:
    - ``sphere`` uses ``center`` / ``radius``
    - ``box`` uses ``box_min`` / ``box_max``
    - ``plane`` uses ``normal`` / ``offset``
    """
    return FixedSurfaceSection(
        mode=mode,
        enabled=enabled,
        axis=axis,
        side=side,
        position=position,
        relative=relative,
        center=center,
        radius=radius,
        box_min=box_min,
        box_max=box_max,
        normal=normal,
        offset=offset,
        value=value,
    )


def fixed_sphere_section(
    *,
    center: tuple[float, ...],
    radius: float,
    value: tuple[float, ...],
    enabled: bool = True,
) -> FixedSurfaceSection:
    """Convenience wrapper for a spherical fixed-region selection."""
    return fixed_surface_section(
        mode="sphere",
        enabled=enabled,
        center=center,
        radius=radius,
        value=value,
    )


def fixed_box_section(
    *,
    box_min: tuple[float, ...],
    box_max: tuple[float, ...],
    value: tuple[float, ...],
    enabled: bool = True,
) -> FixedSurfaceSection:
    """Convenience wrapper for an axis-aligned box fixed-region selection."""
    return fixed_surface_section(
        mode="box",
        enabled=enabled,
        box_min=box_min,
        box_max=box_max,
        value=value,
    )


def fixed_plane_section(
    *,
    normal: tuple[float, ...],
    offset: float,
    value: tuple[float, ...],
    enabled: bool = True,
) -> FixedSurfaceSection:
    """Convenience wrapper for a plane-based fixed-region selection."""
    return fixed_surface_section(
        mode="plane",
        enabled=enabled,
        normal=normal,
        offset=offset,
        value=value,
    )


def body_section(
    *,
    name: str,
    mesh: str | None = None,
    vertices: Any | None = None,
    cells: Any | None = None,
    faces: Any | None = None,
    material: MaterialSection,
    extract: GeometryExtractName = "volume",
    unit: str = "",
    n_refs: int = 0,
    enabled: bool = True,
    is_obstacle: bool = False,
    transformation: TransformationSection | None = None,
    advanced: GeometryAdvancedSection | None = None,
    fixed_surface: FixedSurfaceSection | None = None,
    fixed_surfaces: list[FixedSurfaceSection] | None = None,
    initial_velocity: tuple[float, float] | None = None,
    initial_solution: tuple[float, float] | None = None,
    initial_acceleration: tuple[float, float] | None = None,
) -> BodySection:
    """Create one body entry for the experiment.

    This is the main place where users specify:
    - mesh path, or ``vertices`` + ``cells`` / ``faces``
    - material
    - optional fixed surfaces
    - optional initial conditions such as ``initial_velocity``
    """
    if cells is not None and faces is not None:
        raise ValueError("body_section accepts either cells=... or faces=..., not both")
    if cells is None and faces is not None:
        cells = faces

    uses_mesh_file = isinstance(mesh, str) and mesh.strip() != ""
    uses_array_mesh = vertices is not None or cells is not None

    if uses_mesh_file == uses_array_mesh:
        raise ValueError(
            "body_section requires exactly one geometry source: either mesh='...' "
            "or vertices=... with cells=.../faces=..."
        )
    if uses_array_mesh and (vertices is None or cells is None):
        raise ValueError("array-backed body_section requires both vertices=... and cells=.../faces=...")
    if uses_array_mesh:
        if extract != "volume":
            raise ValueError("array-backed body_section currently supports extract='volume' only")
        if transformation is not None or advanced is not None or n_refs != 0:
            raise ValueError(
                "array-backed body_section does not yet support transformation, advanced, or n_refs"
            )

    surfaces = list(fixed_surfaces or [])
    if fixed_surface is not None:
        surfaces.insert(0, fixed_surface)
    return BodySection(
        name=name,
        mesh=mesh,
        vertices=vertices,
        cells=cells,
        material=material,
        extract=extract,
        unit=unit,
        n_refs=n_refs,
        enabled=enabled,
        is_obstacle=is_obstacle,
        transformation=transformation,
        advanced=advanced,
        fixed_surfaces=surfaces,
        initial_velocity=initial_velocity,
        initial_solution=initial_solution,
        initial_acceleration=initial_acceleration,
    )


def bodies_section(*bodies: BodySection) -> list[BodySection]:
    """Pack one or more body sections into the experiment body list."""
    return list(bodies)


def plane_obstacle_section(
    *,
    point: tuple[float, ...],
    normal: tuple[float, ...],
    enabled: bool = True,
) -> PlaneObstacleSection:
    return PlaneObstacleSection(point=point, normal=normal, enabled=enabled)


def ground_obstacle_section(
    *,
    height: float = 0.0,
    enabled: bool = True,
) -> GroundObstacleSection:
    return GroundObstacleSection(height=height, enabled=enabled)


def mesh_sequence_section(
    *,
    files: list[str],
    fps: int = 1,
    extract: GeometryExtractName = "volume",
    unit: str = "",
    n_refs: int = 0,
    enabled: bool = True,
    is_obstacle: bool = True,
) -> MeshSequenceSection:
    return MeshSequenceSection(
        files=list(files),
        fps=fps,
        extract=extract,
        unit=unit,
        n_refs=n_refs,
        enabled=enabled,
        is_obstacle=is_obstacle,
    )


def transformation_section(
    *,
    translation: tuple[float, ...] = (),
    rotation: tuple[float, ...] = (),
    scale: tuple[float, ...] = (),
    dimensions: float | tuple[float, ...] = 1.0,
    rotation_mode: RotationModeName = "xyz",
) -> TransformationSection:
    """Create a simple geometry transform (translate / rotate / scale)."""
    return TransformationSection(
        translation=translation,
        rotation=rotation,
        scale=scale,
        dimensions=dimensions,
        rotation_mode=rotation_mode,
    )


def geometry_advanced_section(
    *,
    normalize_mesh: bool = False,
    force_linear_geometry: bool = False,
    refinement_location: float = 0.5,
    min_component: int = -1,
) -> GeometryAdvancedSection:
    """Less-common geometry preprocessing knobs for advanced experiments."""
    return GeometryAdvancedSection(
        normalize_mesh=normalize_mesh,
        force_linear_geometry=force_linear_geometry,
        refinement_location=refinement_location,
        min_component=min_component,
    )


def loads_section(*, rhs: tuple[float, float] = (0.0, 0.0)) -> LoadsSection:
    """Create body-force / RHS loading, e.g. gravity in 2D."""
    return LoadsSection(rhs=rhs)


def space_section(
    *,
    discr_order: int | None = 1,
    pressure_discr_order: int | None = 1,
    basis_type: BasisTypeName | None = "Lagrange",
    poly_basis_type: PolyBasisTypeName | None = "MFSHarmonic",
    use_p_ref: bool | None = False,
    bc_method: BoundaryConditionMethodName | None = "sample",
    advanced: dict | None = None,
) -> SpaceSection:
    """Space/discretization choices, including the main ``discr_order`` knob."""
    return SpaceSection(
        discr_order=discr_order,
        pressure_discr_order=pressure_discr_order,
        basis_type=basis_type,
        poly_basis_type=poly_basis_type,
        use_p_ref=use_p_ref,
        bc_method=bc_method,
        advanced=advanced,
    )


def time_section(
    *,
    t0: float = 0.0,
    tend: float = 0.02,
    dt: float = 0.01,
    time_steps: int | None = None,
    integrator: TimeIntegratorName = "ImplicitEuler",
    bdf_steps: int = 1,
    gamma: float = 0.5,
    beta: float = 0.25,
    quasistatic: bool = False,
) -> TimeSection:
    """Create transient time settings.

    Common use:
    - ``t0`` start time
    - ``tend`` end time
    - ``dt`` time step size
    - ``integrator`` time integration scheme
    """
    return TimeSection(
        t0=t0,
        tend=tend,
        dt=dt,
        time_steps=time_steps,
        integrator=integrator,
        bdf_steps=bdf_steps,
        gamma=gamma,
        beta=beta,
        quasistatic=quasistatic,
    )


def linear_solver_section(
    *,
    solver: LinearSolverName = "Eigen::PardisoLDLT",
    precond: LinearPreconditionerName | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
) -> LinearSolverSection:
    """Create the linear-solver block.

    ``solver`` exposes common direct and iterative solver names via IDE
    autocomplete.
    """
    return LinearSolverSection(
        solver=solver,
        precond=precond,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def nonlinear_solver_section(
    *,
    solver: NonlinearSolverName = "Newton",
    tolerance: float = 1e-6,
    grad_norm: float = 0.002,
    x_delta: float | None = None,
    max_iterations: int = 800,
    iterations_per_strategy: int | None = None,
    line_search: LineSearchName | None = None,
    residual_tolerance: float = 100.0,
    history_size: int | None = None,
) -> NonlinearSolverSection:
    """Create the nonlinear-solver block.

    Common knobs:
    - ``solver`` (for example ``"Newton"``)
    - ``grad_norm`` stopping threshold
    - ``max_iterations``
    - ``residual_tolerance``
    - optional ``line_search``
    """
    return NonlinearSolverSection(
        solver=solver,
        tolerance=tolerance,
        grad_norm=grad_norm,
        x_delta=x_delta,
        max_iterations=max_iterations,
        iterations_per_strategy=iterations_per_strategy,
        line_search=line_search,
        residual_tolerance=residual_tolerance,
        history_size=history_size,
    )


def solver_contact_section(
    *,
    barrier_stiffness: BarrierStiffnessName | float = "adaptive",
    initial_barrier_stiffness: float | None = None,
    friction_iterations: int | None = None,
    tangential_adhesion_iterations: int | None = None,
    friction_convergence_tol: float | None = None,
    ccd_broad_phase: CCDBroadPhaseName = "hash_grid",
    ccd_tolerance: float = 1e-6,
    ccd_max_iterations: int = 1_000_000,
) -> SolverContactSection:
    """Create solver-side contact tuning parameters.

    ``barrier_stiffness`` accepts the common guided default ``"adaptive"``
    or an explicit numeric value.
    """
    return SolverContactSection(
        barrier_stiffness=barrier_stiffness,
        initial_barrier_stiffness=initial_barrier_stiffness,
        friction_iterations=friction_iterations,
        tangential_adhesion_iterations=tangential_adhesion_iterations,
        friction_convergence_tol=friction_convergence_tol,
        ccd_broad_phase=ccd_broad_phase,
        ccd_tolerance=ccd_tolerance,
        ccd_max_iterations=ccd_max_iterations,
    )


def solver_section(
    *,
    linear: LinearSolverSection | None = None,
    nonlinear: NonlinearSolverSection | None = None,
    contact: SolverContactSection | None = None,
    linear_solver: LinearSolverName | None = None,
    max_iterations: int | None = None,
    grad_norm: float | None = None,
    residual_tolerance: float | None = None,
    barrier_stiffness: BarrierStiffnessName | float | None = None,
) -> SolverSection:
    """Create a solver section.

    The explicit ``linear/nonlinear/contact`` sections are the preferred shape.
    The flat keywords are kept as a convenience bridge from the earlier example.
    """
    linear_section = linear_solver_section() if linear is None else linear
    nonlinear_section = nonlinear_solver_section() if nonlinear is None else nonlinear
    contact_section_obj = solver_contact_section() if contact is None else contact

    if linear_solver is not None:
        linear_section.solver = linear_solver
    if max_iterations is not None:
        nonlinear_section.max_iterations = max_iterations
    if grad_norm is not None:
        nonlinear_section.grad_norm = grad_norm
    if residual_tolerance is not None:
        nonlinear_section.residual_tolerance = residual_tolerance
    if barrier_stiffness is not None:
        contact_section_obj.barrier_stiffness = barrier_stiffness

    return SolverSection(
        linear=linear_section,
        nonlinear=nonlinear_section,
        contact=contact_section_obj,
    )


def contact_section(
    *,
    mode: ContactModeName = "disabled",
    dhat: float = 0.001,
    mu: float = 0.0,
    dhat_percentage: float = 0.8,
    epsv: float = 0.001,
    use_convergent_formulation: bool = False,
    use_area_weighting: bool = True,
    use_improved_max_operator: bool = True,
    use_physical_barrier: bool = True,
    use_gcp_formulation: bool = False,
    alpha_n: float = 0.5,
    alpha_t: float = 0.5,
    min_distance_ratio: float = 0.5,
    use_adaptive_dhat: bool = False,
    periodic: bool = False,
    barrier_stiffness: BarrierStiffnessName | float | None = None,
    collision_mesh: CollisionMeshSection | None = None,
    adhesion: AdhesionSection | None = None,
    adhesion_strength: float = 0.001,
) -> ContactSection:
    """Create the physical contact settings.

    Common modes exposed through IDE autocomplete:
    - ``"disabled"``
    - ``"frictionless"``
    - ``"coulomb"``
    - ``"adhesive"``
    """
    return ContactSection(
        mode=mode,
        dhat=dhat,
        mu=mu,
        dhat_percentage=dhat_percentage,
        epsv=epsv,
        use_convergent_formulation=use_convergent_formulation,
        use_area_weighting=use_area_weighting,
        use_improved_max_operator=use_improved_max_operator,
        use_physical_barrier=use_physical_barrier,
        use_gcp_formulation=use_gcp_formulation,
        alpha_n=alpha_n,
        alpha_t=alpha_t,
        min_distance_ratio=min_distance_ratio,
        use_adaptive_dhat=use_adaptive_dhat,
        periodic=periodic,
        barrier_stiffness=barrier_stiffness,
        collision_mesh=collision_mesh,
        adhesion=adhesion,
        adhesion_strength=adhesion_strength,
    )


def collision_mesh_section(
    *,
    enabled: bool = True,
    tessellation_type: CollisionTessellationTypeName = "regular",
    mesh: str | None = None,
    linear_map: str | None = None,
    max_edge_length: float | None = None,
) -> CollisionMeshSection:
    return CollisionMeshSection(
        enabled=enabled,
        tessellation_type=tessellation_type,
        mesh=mesh,
        linear_map=linear_map,
        max_edge_length=max_edge_length,
    )


def adhesion_section(
    *,
    enabled: bool = False,
    dhat_p: float = 0.001,
    dhat_a: float = 0.01,
    adhesion_strength: float = 0.001,
    tangential_adhesion_coefficient: float = 0.0,
    epsa: float = 0.001,
) -> AdhesionSection:
    return AdhesionSection(
        enabled=enabled,
        dhat_p=dhat_p,
        dhat_a=dhat_a,
        adhesion_strength=adhesion_strength,
        tangential_adhesion_coefficient=tangential_adhesion_coefficient,
        epsa=epsa,
    )


def results_section(
    *,
    requested_fields: list[ResultFieldName | str] | None = None,
    strict: bool = False,
) -> ResultsSection:
    """Request Python-side result/history fields.

    Default fields are:
    - ``"u"``
    - ``"stress"``
    - ``"von_mises"``

    Common IDE-suggested choices also include:
    - ``"strain"``
    - ``"pressure"``
    - ``"body_ids"``
    - ``"velocity"``
    - ``"acceleration"``
    """
    return ResultsSection(
        requested_fields=list(DEFAULT_REQUESTED_FIELDS) if requested_fields is None else list(requested_fields),
        strict=strict,
    )


def output_files_section(
    *,
    save_vtu: bool = False,
    json_name: str = "",
    restart_json_name: str = "",
) -> OutputFilesSection:
    """Configure simple file outputs such as VTU and JSON filenames."""
    return OutputFilesSection(
        save_vtu=save_vtu,
        json_name=json_name,
        restart_json_name=restart_json_name,
    )


def output_log_section(
    *,
    level: LogLevelName = "debug",
    file_level: LogLevelName = "trace",
    path: str = "",
    quiet: bool = False,
) -> OutputLogSection:
    """Configure terminal/file logging for the legacy output section path."""
    return OutputLogSection(
        level=level,
        file_level=file_level,
        path=path,
        quiet=quiet,
    )


def paraview_fields_section(
    *,
    use_hdf5: bool = False,
    material: bool = False,
    body_ids: bool = False,
    contact_forces: bool = False,
    friction_forces: bool = False,
    normal_adhesion_forces: bool = False,
    tangential_adhesion_forces: bool = False,
    velocity: bool = False,
    acceleration: bool = False,
    scalar_values: bool = True,
    tensor_values: bool = True,
    discretization_order: bool = True,
    nodes: bool = True,
    forces: bool = False,
    force_high_order: bool = False,
    jacobian_validity: bool = False,
) -> ParaviewFieldsSection:
    """Choose which fields should be written into ParaView outputs."""
    return ParaviewFieldsSection(
        use_hdf5=use_hdf5,
        material=material,
        body_ids=body_ids,
        contact_forces=contact_forces,
        friction_forces=friction_forces,
        normal_adhesion_forces=normal_adhesion_forces,
        tangential_adhesion_forces=tangential_adhesion_forces,
        velocity=velocity,
        acceleration=acceleration,
        scalar_values=scalar_values,
        tensor_values=tensor_values,
        discretization_order=discretization_order,
        nodes=nodes,
        forces=forces,
        force_high_order=force_high_order,
        jacobian_validity=jacobian_validity,
    )


def paraview_section(
    *,
    file_name: str = "",
    volume: bool = True,
    surface: bool = False,
    wireframe: bool = False,
    points: bool = False,
    vismesh_rel_area: float = 1e-5,
    skip_frame: int = 1,
    high_order_mesh: bool = True,
    fields: list[str] | None = None,
    options: ParaviewFieldsSection | None = None,
) -> ParaviewSection:
    """Configure ParaView sequence output (PVD + VTU settings)."""
    return ParaviewSection(
        file_name=file_name,
        volume=volume,
        surface=surface,
        wireframe=wireframe,
        points=points,
        vismesh_rel_area=vismesh_rel_area,
        skip_frame=skip_frame,
        high_order_mesh=high_order_mesh,
        fields=[] if fields is None else list(fields),
        options=paraview_fields_section() if options is None else options,
    )


def output_data_section(
    *,
    solution: str = "",
    full_mat: str = "",
    stiffness_mat: str = "",
    stress_mat: str = "",
    state: str = "",
    rest_mesh: str = "",
    mises: str = "",
    nodes: str = "",
    reorder_nodes: bool = False,
    file_index_offset: int = 0,
) -> OutputDataSection:
    """Configure low-level matrix/state dumps written by the solver."""
    return OutputDataSection(
        solution=solution,
        full_mat=full_mat,
        stiffness_mat=stiffness_mat,
        stress_mat=stress_mat,
        state=state,
        rest_mesh=rest_mesh,
        mises=mises,
        nodes=nodes,
        reorder_nodes=reorder_nodes,
        file_index_offset=file_index_offset,
    )


def output_reference_section(
    *,
    solution: list[str] | None = None,
    gradient: list[str] | None = None,
) -> OutputReferenceSection:
    """Configure optional reference solution / gradient files for comparisons."""
    return OutputReferenceSection(
        solution=[] if solution is None else list(solution),
        gradient=[] if gradient is None else list(gradient),
    )


def output_advanced_section(
    *,
    timestep_prefix: str = "step_",
    sol_on_grid: float = -1.0,
    compute_error: bool = True,
    sol_at_node: int = -1,
    vis_boundary_only: bool = False,
    curved_mesh_size: bool = False,
    save_solve_sequence_debug: bool = False,
    save_ccd_debug_meshes: bool = False,
    save_time_sequence: bool = True,
    save_nl_solve_sequence: bool = False,
    spectrum: bool = False,
) -> OutputAdvancedSection:
    """Configure advanced output behavior such as time-sequence export."""
    return OutputAdvancedSection(
        timestep_prefix=timestep_prefix,
        sol_on_grid=sol_on_grid,
        compute_error=compute_error,
        sol_at_node=sol_at_node,
        vis_boundary_only=vis_boundary_only,
        curved_mesh_size=curved_mesh_size,
        save_solve_sequence_debug=save_solve_sequence_debug,
        save_ccd_debug_meshes=save_ccd_debug_meshes,
        save_time_sequence=save_time_sequence,
        save_nl_solve_sequence=save_nl_solve_sequence,
        spectrum=spectrum,
    )


def output_section(
    *,
    directory: str = "",
    stats: bool = False,
    files: OutputFilesSection | None = None,
    log: OutputLogSection | None = None,
    paraview: ParaviewSection | None = None,
    advanced: OutputAdvancedSection | None = None,
    data: OutputDataSection | None = None,
    reference: OutputReferenceSection | None = None,
    paraview_fields: ParaviewFieldsSection | None = None,
    vismesh_rel_area: float | None = None,
) -> OutputSection:
    """Create an output section.

    ``paraview_fields`` / ``vismesh_rel_area`` are accepted for compatibility
    with the earlier guided-template shape and are folded into ``paraview``.

    Most users can avoid this lower-level helper in examples by calling the
    runtime helpers ``terminal_log(cfg)`` and ``result_output(cfg)`` after
    ``build_config(...)``.
    """
    paraview_section_obj = paraview_section() if paraview is None else paraview
    if paraview_fields is not None:
        paraview_section_obj.options = paraview_fields
    if vismesh_rel_area is not None:
        paraview_section_obj.vismesh_rel_area = vismesh_rel_area

    return OutputSection(
        directory=directory,
        stats=stats,
        files=output_files_section() if files is None else files,
        log=output_log_section() if log is None else log,
        paraview=paraview_section_obj,
        advanced=output_advanced_section() if advanced is None else advanced,
        data=output_data_section() if data is None else data,
        reference=output_reference_section() if reference is None else reference,
    )


def experiment_template(
    *,
    problem: ProblemSection | None = None,
    units: UnitsSection | None = None,
    bodies: list[BodySection] | None = None,
    space: SpaceSection | None = None,
    geometry_extras: list[PlaneObstacleSection | GroundObstacleSection | MeshSequenceSection] | None = None,
    loads: LoadsSection | None = None,
    time: TimeSection | None = None,
    solver: SolverSection | None = None,
    contact: ContactSection | None = None,
    results: ResultsSection | None = None,
    output: OutputSection | None = None,
) -> ExperimentTemplate:
    """Assemble the full guided experiment template.

    This is the top-level container that gathers all sections before
    ``build_config(...)`` turns them into a concrete ``SimulationConfig``.
    """
    return ExperimentTemplate(
        problem=problem_section() if problem is None else problem,
        units=units_section() if units is None else units,
        bodies=[] if bodies is None else list(bodies),
        space=space_section() if space is None else space,
        geometry_extras=[] if geometry_extras is None else list(geometry_extras),
        loads=loads_section() if loads is None else loads,
        time=time_section() if time is None else time,
        solver=solver_section() if solver is None else solver,
        contact=contact_section() if contact is None else contact,
        results=results_section() if results is None else results,
        output=output_section() if output is None else output,
    )


def impact_template(**kwargs) -> ExperimentTemplate:
    """Backward-compatible preset helper for the old impact example flow."""
    template = default_impact_template()

    lattice = kwargs.pop("lattice", None)
    block = kwargs.pop("block", None)
    bodies = kwargs.pop("bodies", None)
    if bodies is not None:
        template.bodies = list(bodies)
    else:
        current_bodies = list(template.bodies)
        if lattice is not None:
            if current_bodies:
                current_bodies[0] = lattice
            else:
                current_bodies.append(lattice)
        if block is not None:
            if len(current_bodies) >= 2:
                current_bodies[1] = block
            else:
                while len(current_bodies) < 1:
                    current_bodies.append(body_section(name="body", mesh="", material=material_section()))
                current_bodies.append(block)
        template.bodies = current_bodies

    for key, value in kwargs.items():
        if not hasattr(template, key):
            raise TypeError(f"impact_template() got an unexpected keyword argument {key!r}")
        setattr(template, key, value)
    return template


def build_material(section: MaterialSection):
    if section.mode == "young_poisson":
        if section.model == "NeoHookean":
            return NeoHookean.young_poisson(
                E=section.E,
                E_unit=section.E_unit,
                nu=section.nu,
                rho=section.rho,
                rho_unit=section.rho_unit,
            )
        if section.model == "LinearElasticity":
            return LinearElasticity.young_poisson(
                E=section.E,
                E_unit=section.E_unit,
                nu=section.nu,
                rho=section.rho,
                rho_unit=section.rho_unit,
            )
        if section.model == "SaintVenant":
            return SaintVenant.young_poisson(
                E=section.E,
                E_unit=section.E_unit,
                nu=section.nu,
                rho=section.rho,
                rho_unit=section.rho_unit,
            )
    if section.mode == "lame":
        if section.lambda_ is None or section.mu is None:
            raise ValueError("lame mode requires lambda_ and mu")
        if section.model == "NeoHookean":
            return NeoHookean.lame(
                lambda_=section.lambda_,
                lambda_unit=section.lambda_unit,
                mu=section.mu,
                mu_unit=section.mu_unit,
                rho=section.rho,
                rho_unit=section.rho_unit,
            )
        if section.model == "LinearElasticity":
            return LinearElasticity.lame(
                lambda_=section.lambda_,
                lambda_unit=section.lambda_unit,
                mu=section.mu,
                mu_unit=section.mu_unit,
                rho=section.rho,
                rho_unit=section.rho_unit,
            )
    raise ValueError(f"unsupported material mode: {section.mode!r}")


def add_body_from_section(cfg: SimulationConfig, section: BodySection):
    if section.mesh is not None:
        geometry = GeometryMesh.from_file(
            mesh_file(section.mesh),
            extract=section.extract,
            unit=section.unit,
            transformation=GeometryTransformation(
                translation=list(section.transformation.translation),
                rotation=list(section.transformation.rotation),
                scale=list(section.transformation.scale),
                dimensions=section.transformation.dimensions,
                rotation_mode=section.transformation.rotation_mode,
            )
            if section.transformation is not None
            else None,
            n_refs=section.n_refs,
            advanced=GeometryAdvanced(
                normalize_mesh=section.advanced.normalize_mesh,
                force_linear_geometry=section.advanced.force_linear_geometry,
                refinement_location=section.advanced.refinement_location,
                min_component=section.advanced.min_component,
            )
            if section.advanced is not None
            else None,
            enabled=section.enabled,
            is_obstacle=section.is_obstacle,
        )
    else:
        geometry = GeometryMesh(
            mesh=f"__array_body__:{section.name}",
            extract=section.extract,
            unit=section.unit,
            enabled=section.enabled,
            is_obstacle=section.is_obstacle,
        )

    body = cfg.add_body(
        name=section.name,
        geometry=geometry,
        material=build_material(section.material),
    )
    for surface in section.fixed_surfaces:
        if surface.enabled:
            body.fix_surface(build_surface_selection(surface), value=list(surface.value))
    if section.initial_velocity is not None:
        body.set_initial_velocity(list(section.initial_velocity))
    if section.initial_solution is not None:
        body.set_initial_solution(list(section.initial_solution))
    if section.initial_acceleration is not None:
        body.set_initial_acceleration(list(section.initial_acceleration))
    return body


def _is_array_backed_body(section: BodySection) -> bool:
    return section.vertices is not None or section.cells is not None


def _coerce_body_vertices(vertices: Any, *, body_name: str) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float64)
    if vertices_np.ndim != 2:
        raise ValueError(
            f"array-backed body '{body_name}' requires vertices with shape (n_vertices, dim), "
            f"got {vertices_np.shape!r}"
        )
    if vertices_np.shape[0] == 0:
        raise ValueError(f"array-backed body '{body_name}' requires at least one vertex")
    return vertices_np


def _coerce_body_cells(cells: Any, *, body_name: str) -> np.ndarray:
    cells_np = np.asarray(cells, dtype=np.int32)
    if cells_np.ndim != 2:
        raise ValueError(
            f"array-backed body '{body_name}' requires cells/faces with shape (n_cells, k), "
            f"got {cells_np.shape!r}"
        )
    if cells_np.shape[0] == 0:
        raise ValueError(f"array-backed body '{body_name}' requires at least one cell/face")
    return cells_np


def _build_guided_array_mesh_payload(
    array_bodies: list[tuple[BodySection, Any]],
) -> dict[str, np.ndarray]:
    merged_vertices: list[np.ndarray] = []
    merged_cells: list[np.ndarray] = []
    merged_body_ids: list[np.ndarray] = []

    expected_dim: int | None = None
    expected_cell_width: int | None = None
    vertex_offset = 0

    for section, body in array_bodies:
        vertices_np = _coerce_body_vertices(section.vertices, body_name=section.name)
        cells_np = _coerce_body_cells(section.cells, body_name=section.name)

        if expected_dim is None:
            expected_dim = int(vertices_np.shape[1])
        elif vertices_np.shape[1] != expected_dim:
            raise ValueError(
                "all array-backed bodies must use the same vertex dimension; "
                f"expected {expected_dim}, got {vertices_np.shape[1]} for body '{section.name}'"
            )

        if expected_cell_width is None:
            expected_cell_width = int(cells_np.shape[1])
        elif cells_np.shape[1] != expected_cell_width:
            raise ValueError(
                "all array-backed bodies must use the same cell width; "
                f"expected {expected_cell_width}, got {cells_np.shape[1]} for body '{section.name}'"
            )

        if np.any(cells_np < 0):
            raise ValueError(f"array-backed body '{section.name}' contains negative cell indices")
        if int(cells_np.max()) >= int(vertices_np.shape[0]):
            raise ValueError(
                f"array-backed body '{section.name}' has cell indices outside its vertex range"
            )

        merged_vertices.append(vertices_np)
        merged_cells.append(cells_np + vertex_offset)
        merged_body_ids.append(np.full(cells_np.shape[0], body.volume_id, dtype=np.int32))
        vertex_offset += int(vertices_np.shape[0])

    return {
        "vertices": np.vstack(merged_vertices),
        "cells": np.vstack(merged_cells),
        "body_ids": np.concatenate(merged_body_ids),
    }


def build_surface_selection(section: FixedSurfaceSection) -> SurfaceSelection:
    side_to_axis = {
        "x_min": -1,
        "x_max": 1,
        "y_min": -2,
        "y_max": 2,
        "z_min": -3,
        "z_max": 3,
    }

    if section.mode == "position":
        axis = side_to_axis[section.side] if section.side is not None else section.axis
        return SurfaceSelection.position(
            axis=axis,
            position=section.position,
            relative=section.relative,
        )
    if section.mode == "sphere":
        if section.center is None or section.radius is None:
            raise ValueError("sphere fixed surface requires center and radius")
        return SurfaceSelection.sphere(
            center=list(section.center),
            radius=section.radius,
        )
    if section.mode == "box":
        if section.box_min is None or section.box_max is None:
            raise ValueError("box fixed surface requires box_min and box_max")
        return SurfaceSelection.box(
            box_min=list(section.box_min),
            box_max=list(section.box_max),
        )
    if section.mode == "plane":
        if section.normal is None or section.offset is None:
            raise ValueError("plane fixed surface requires normal and offset")
        return SurfaceSelection.plane(
            normal=list(section.normal),
            offset=section.offset,
        )
    raise ValueError(f"unsupported fixed surface mode: {section.mode!r}")


def build_space(section: SpaceSection) -> Space:
    advanced = dict(section.advanced or {})
    if section.bc_method is not None:
        advanced["bc_method"] = section.bc_method

    polynomial_type = section.basis_type
    if polynomial_type is None:
        polynomial_type = section.poly_basis_type

    return Space(
        discr_order=section.discr_order,
        pressure_discr_order=section.pressure_discr_order,
        use_p_ref=section.use_p_ref,
        polynomial_type=polynomial_type,
        advanced=advanced or None,
    )


def build_geometry_extra(section: PlaneObstacleSection | GroundObstacleSection | MeshSequenceSection):
    if isinstance(section, PlaneObstacleSection):
        return GeometryPlane.obstacle(
            point=list(section.point),
            normal=list(section.normal),
            enabled=section.enabled,
        )
    if isinstance(section, GroundObstacleSection):
        return GeometryGround.obstacle(
            height=section.height,
            enabled=section.enabled,
        )
    if isinstance(section, MeshSequenceSection):
        return GeometryMeshSequence.from_files(
            list(section.files),
            fps=section.fps,
            unit=section.unit,
            extract=section.extract,
            n_refs=section.n_refs,
            enabled=section.enabled,
            is_obstacle=section.is_obstacle,
        )
    raise TypeError(f"Unsupported geometry extra section: {type(section).__name__}")


def build_time(section: TimeSection) -> Time:
    if section.integrator == "ImplicitNewmark":
        return Time.implicit_newmark(
            t0=section.t0,
            tend=section.tend,
            dt=section.dt,
            time_steps=section.time_steps,
            gamma=section.gamma,
            beta=section.beta,
            quasistatic=section.quasistatic,
        )
    if section.integrator.startswith("BDF"):
        steps = section.bdf_steps
        if section.integrator != "BDF1":
            try:
                steps = int(section.integrator.replace("BDF", ""))
            except ValueError:
                steps = section.bdf_steps
        return Time.bdf(
            t0=section.t0,
            tend=section.tend,
            dt=section.dt,
            time_steps=section.time_steps,
            steps=steps,
            quasistatic=section.quasistatic,
        )
    return Time.transient(
        t0=section.t0,
        tend=section.tend,
        dt=section.dt,
        time_steps=section.time_steps,
        integrator=section.integrator,
        quasistatic=section.quasistatic,
    )


def build_solver(section: SolverSection) -> Solver:
    linear = LinearSolver(
        solver_type=section.linear.solver,
        precond=section.linear.precond,
        max_iterations=section.linear.max_iterations,
        tolerance=section.linear.tolerance,
    )

    method_blocks = None
    if section.nonlinear.solver == "Newton":
        method_blocks = {"Newton": {"residual_tolerance": section.nonlinear.residual_tolerance}}
    elif section.nonlinear.solver in {"L-BFGS", "L-BFGS-B"} and section.nonlinear.history_size is not None:
        method_blocks = {section.nonlinear.solver: {"history_size": section.nonlinear.history_size}}

    nonlinear = NonlinearSolver(
        solver_type=section.nonlinear.solver,
        max_iterations=section.nonlinear.max_iterations,
        tolerance=section.nonlinear.tolerance,
        grad_norm=section.nonlinear.grad_norm,
        x_delta=section.nonlinear.x_delta,
        iterations_per_strategy=section.nonlinear.iterations_per_strategy,
        line_search=section.nonlinear.line_search,
        method_blocks=method_blocks,
    )

    contact = SolverContactOptions(
        CCD={
            "broad_phase": section.contact.ccd_broad_phase,
            "tolerance": section.contact.ccd_tolerance,
            "max_iterations": section.contact.ccd_max_iterations,
        },
        friction_iterations=section.contact.friction_iterations,
        tangential_adhesion_iterations=section.contact.tangential_adhesion_iterations,
        friction_convergence_tol=section.contact.friction_convergence_tol,
        barrier_stiffness=section.contact.barrier_stiffness,
        initial_barrier_stiffness=section.contact.initial_barrier_stiffness,
    )

    return Solver(
        linear=linear,
        nonlinear=nonlinear,
        contact=contact,
    )


def build_contact(section: ContactSection) -> Contact | None:
    if section.mode == "disabled":
        return None

    collision_mesh = None
    if section.collision_mesh is not None:
        collision_mesh = CollisionMesh(
            enabled=section.collision_mesh.enabled,
            tessellation_type=section.collision_mesh.tessellation_type,
            mesh=section.collision_mesh.mesh,
            linear_map=section.collision_mesh.linear_map,
            max_edge_length=section.collision_mesh.max_edge_length,
        )

    adhesion = None
    if section.adhesion is not None:
        adhesion = Adhesion(
            adhesion_enabled=section.adhesion.enabled,
            dhat_p=section.adhesion.dhat_p,
            dhat_a=section.adhesion.dhat_a,
            adhesion_strength=section.adhesion.adhesion_strength,
            tangential_adhesion_coefficient=section.adhesion.tangential_adhesion_coefficient,
            epsa=section.adhesion.epsa,
        )
    elif section.mode == "adhesive":
        adhesion = Adhesion(
            adhesion_enabled=True,
            adhesion_strength=section.adhesion_strength,
        )

    return Contact(
        enabled=True,
        dhat=section.dhat,
        dhat_percentage=section.dhat_percentage,
        epsv=section.epsv,
        friction_coefficient=section.mu,
        mu=section.mu,
        use_convergent_formulation=section.use_convergent_formulation,
        use_area_weighting=section.use_area_weighting,
        use_improved_max_operator=section.use_improved_max_operator,
        use_physical_barrier=section.use_physical_barrier,
        collision_mesh=collision_mesh,
        use_gcp_formulation=section.use_gcp_formulation,
        alpha_n=section.alpha_n,
        alpha_t=section.alpha_t,
        min_distance_ratio=section.min_distance_ratio,
        use_adaptive_dhat=section.use_adaptive_dhat,
        periodic=section.periodic,
        adhesion=adhesion,
        barrier_stiffness=section.barrier_stiffness,
    )


def build_output(section: OutputSection, results: ResultsSection, workspace: Path) -> Output:
    output_dir = Path(section.directory)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()

    json_target: bool | str = False
    if section.files.json_name:
        json_target = section.files.json_name

    output = Output(
        directory=str(output_dir),
        json=json_target,
        restart_json=section.files.restart_json_name or None,
        log=OutputLog(
            level=section.log.level,
            file_level=section.log.file_level,
            path=section.log.path,
            quiet=section.log.quiet,
        ),
        paraview=ParaviewOutput(
            volume=section.paraview.volume,
            surface=section.paraview.surface,
            wireframe=section.paraview.wireframe,
            points=section.paraview.points,
            file_name=section.paraview.file_name or None,
            options=OutputParaviewOptions(
                use_hdf5=section.paraview.options.use_hdf5,
                material=section.paraview.options.material,
                body_ids=section.paraview.options.body_ids,
                contact_forces=section.paraview.options.contact_forces,
                friction_forces=section.paraview.options.friction_forces,
                normal_adhesion_forces=section.paraview.options.normal_adhesion_forces,
                tangential_adhesion_forces=section.paraview.options.tangential_adhesion_forces,
                velocity=section.paraview.options.velocity,
                acceleration=section.paraview.options.acceleration,
                scalar_values=section.paraview.options.scalar_values,
                tensor_values=section.paraview.options.tensor_values,
                discretization_order=section.paraview.options.discretization_order,
                nodes=section.paraview.options.nodes,
                forces=section.paraview.options.forces,
                force_high_order=section.paraview.options.force_high_order,
                jacobian_validity=section.paraview.options.jacobian_validity,
            ),
            vismesh_rel_area=section.paraview.vismesh_rel_area,
            skip_frame=section.paraview.skip_frame,
            high_order_mesh=section.paraview.high_order_mesh,
            fields=list(section.paraview.fields),
        ),
        data=OutputData(
            solution=section.data.solution,
            full_mat=section.data.full_mat,
            stiffness_mat=section.data.stiffness_mat,
            stress_mat=section.data.stress_mat,
            state=section.data.state,
            rest_mesh=section.data.rest_mesh,
            mises=section.data.mises,
            nodes=section.data.nodes,
            advanced=OutputDataAdvanced(reorder_nodes=section.data.reorder_nodes),
            file_index_offset=section.data.file_index_offset,
        ),
        advanced=OutputAdvanced(
            timestep_prefix=section.advanced.timestep_prefix,
            sol_on_grid=section.advanced.sol_on_grid,
            compute_error=section.advanced.compute_error,
            sol_at_node=section.advanced.sol_at_node,
            vis_boundary_only=section.advanced.vis_boundary_only,
            curved_mesh_size=section.advanced.curved_mesh_size,
            save_solve_sequence_debug=section.advanced.save_solve_sequence_debug,
            save_ccd_debug_meshes=section.advanced.save_ccd_debug_meshes,
            save_time_sequence=section.advanced.save_time_sequence,
            save_nl_solve_sequence=section.advanced.save_nl_solve_sequence,
            spectrum=section.advanced.spectrum,
        ),
        reference=OutputReference(
            solution=list(section.reference.solution),
            gradient=list(section.reference.gradient),
        ),
        stats=section.stats,
    )
    output.save_vtu = section.files.save_vtu
    output.request_results(list(results.requested_fields), strict=results.strict)
    output.resolve_relative_paths(output_dir)
    return output


def build_config(template: ExperimentTemplate, workspace: Path) -> SimulationConfig:
    if not template.bodies:
        raise ValueError("guided template requires at least one body in template.bodies")

    cfg = SimulationConfig()
    cfg.pde = template.problem.pde
    cfg.problem_type = template.problem.problem_type
    cfg.problem_params = template.problem.problem_params
    cfg.discr_order = 1 if template.space.discr_order is None else template.space.discr_order
    cfg.units = Units.set_units(
        length=template.units.length,
        mass=template.units.mass,
        time=template.units.time,
        characteristic_length=template.units.characteristic_length,
    )

    array_bodies: list[tuple[BodySection, Any]] = []
    file_backed_body_seen = False

    for body_section_obj in template.bodies:
        body = add_body_from_section(cfg, body_section_obj)
        if _is_array_backed_body(body_section_obj):
            array_bodies.append((body_section_obj, body))
        else:
            file_backed_body_seen = True

    if array_bodies:
        if file_backed_body_seen:
            raise ValueError(
                "guided templates cannot currently mix mesh-file bodies with vertices/cells bodies"
            )
        cfg.extras["_mesh_array_mode"] = _build_guided_array_mesh_payload(array_bodies)

    if template.geometry_extras:
        geometry_obj = cfg._ensure_geometry_object()
        for extra in template.geometry_extras:
            geometry_obj.add(build_geometry_extra(extra))

    cfg.set_rhs(list(template.loads.rhs))
    cfg.solver = build_solver(template.solver)
    cfg.time = build_time(template.time)
    cfg.space = build_space(template.space)
    cfg.output = build_output(template.output, template.results, workspace)
    cfg.contact = build_contact(template.contact)
    return cfg
