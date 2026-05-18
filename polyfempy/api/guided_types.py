"""Section dataclasses and type aliases for the guided config API.

This module is intentionally data-only.  User-facing factory functions and
``build_config(...)`` live in ``guided_sections.py`` so existing imports keep
working while the section schema is easier to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


DEFAULT_REQUESTED_FIELDS = ("u", "stress", "von_mises")


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


SimulationTemplate = ExperimentTemplate
