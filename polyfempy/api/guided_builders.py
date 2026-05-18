"""User-facing factory functions for guided config sections.

These helpers create the typed section objects used by ``polyfempy.api.guided``.
They intentionally do not build ``SimulationConfig`` objects;
``guided_sections.build_config`` owns that lower-level translation.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from .guided_types import (
    DEFAULT_REQUESTED_FIELDS,
    AdhesionSection,
    AxisSideName,
    BarrierStiffnessName,
    BasisTypeName,
    BodySection,
    BoundaryConditionMethodName,
    CCDBroadPhaseName,
    CollisionMeshSection,
    CollisionTessellationTypeName,
    ContactModeName,
    ContactSection,
    DensityUnitName,
    ExperimentTemplate,
    FixedSurfaceSection,
    GeometryAdvancedSection,
    GeometryExtractName,
    GroundObstacleSection,
    LameModelName,
    LengthUnitName,
    LinearPreconditionerName,
    LinearSolverName,
    LinearSolverSection,
    LoadsSection,
    LogLevelName,
    MassUnitName,
    MaterialModeName,
    MaterialModelName,
    MaterialSection,
    MeshSequenceSection,
    NonlinearSolverName,
    NonlinearSolverSection,
    OutputAdvancedSection,
    OutputDataSection,
    OutputFilesSection,
    OutputLogSection,
    OutputReferenceSection,
    OutputSection,
    PDEName,
    ParaviewFieldsSection,
    ParaviewSection,
    PlaneObstacleSection,
    PolyBasisTypeName,
    PressureUnitName,
    ProblemSection,
    ProblemTypeName,
    ResultFieldName,
    ResultsSection,
    RotationModeName,
    SimulationTemplate,
    SolverContactSection,
    SolverSection,
    SpaceSection,
    SurfaceSelectionModeName,
    TimeIntegratorName,
    TimeSection,
    TimeUnitName,
    TransformationSection,
    UnitsSection,
    YoungPoissonModelName,
    LineSearchName,
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


def simulation_template(
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
) -> SimulationTemplate:
    """Assemble a guided simulation template.

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


def experiment_template(**kwargs) -> ExperimentTemplate:
    """Compatibility alias for ``simulation_template(...)``."""
    return simulation_template(**kwargs)
