"""Guided template to ``SimulationConfig`` translation.

This internal module owns the lower-level conversion from user-authored guided
section objects into a solver-facing ``SimulationConfig``. Public section
factories live in ``guided_builders.py`` and the public facade lives in
``guided.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
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
from ._guided_array_mesh import (
    build_guided_array_mesh_payload,
    is_array_backed_body,
)
from .guided_types import (
    BodySection,
    ContactSection,
    FixedSurfaceSection,
    GroundObstacleSection,
    MaterialSection,
    MeshSequenceSection,
    OutputSection,
    PlaneObstacleSection,
    ResultsSection,
    SimulationTemplate,
    SolverSection,
    SpaceSection,
    TimeSection,
)


def mesh_file(name: str) -> str:
    path = Path(name)
    return str(path if path.is_absolute() else path.resolve())


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


def build_config(template: SimulationTemplate, workspace: Path) -> SimulationConfig:
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
        if is_array_backed_body(body_section_obj):
            array_bodies.append((body_section_obj, body))
        else:
            file_backed_body_seen = True

    if array_bodies:
        if file_backed_body_seen:
            raise ValueError(
                "guided templates cannot currently mix mesh-file bodies with vertices/cells bodies"
            )
        cfg.extras["_mesh_array_mode"] = build_guided_array_mesh_payload(array_bodies)

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


__all__ = [
    "add_body_from_section",
    "build_config",
    "build_contact",
    "build_geometry_extra",
    "build_material",
    "build_output",
    "build_solver",
    "build_space",
    "build_surface_selection",
    "build_time",
    "mesh_file",
]
