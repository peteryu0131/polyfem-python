# API Class Roadmap

This note compares the current hand-written `polyfempy.api.config` classes
against the PolyFEM JSON schema (`input-spec.json`) and the mechanically
generated class skeleton (`generated_class.py`).

Use it as a development guide for deciding:
- which schema blocks already have a good public Python class
- which ones are only partially typed and still hide `dict` islands
- which top-level blocks are still missing
- which defaults from the schema are worth carrying into future typed classes

## Source Of Truth

- Primary source of semantics/defaults: `input-spec.json`
- Secondary source for nested structure coverage: `generated_class.py`
- Public API style source: `polyfempy/api/config.py`

In practice:
- use `input-spec.json` to decide field names, required/optional status, enums, defaults
- use `generated_class.py` only to discover coverage and nesting
- do not copy `Object1` / `Value2` / `__2` naming into the public API

## Current Status

### Already Solid

These areas already have useful, hand-written, public-facing classes.

| Area | Current classes | Notes |
| --- | --- | --- |
| Root config | `SimulationConfig` | Good public entry point already exists. |
| Units/value wrappers | `Units`, `Quantity` | Good Python-first direction. |
| Materials | `NeoHookean`, `LinearElasticity`, `HookeLinearElasticity`, `SaintVenant`, `MooneyRivlin`, `Ogden`, `Stokes`, `NavierStokes`, etc. | Best-covered part of the API today. |
| Basic geometry | `Geometry`, `GeometryMesh` | Good start, but only one geometry variant is typed. |
| Basic solver | `LinearSolver`, `NonlinearSolver`, `Solver` | Useful top-level entry point exists. |
| Basic time | `Time` | Good minimal class, but not all integrator variants are typed. |
| Basic output | `ParaviewOutput`, `Output` | Useful top-level entry point exists. |
| Basic BCs | `DirichletBoundary`, `NeumannBoundary`, `BoundaryConditions` | Covers common cases. |
| Basic contact | `Contact` | Covers only a simplified subset of the schema. |

### Partially Typed

These blocks exist at the top level, but still rely heavily on raw `dict`s,
`Union[..., Dict[str, Any]]`, or simplified subsets of the real schema.

| Area | Current state | What is still raw / simplified |
| --- | --- | --- |
| Geometry | `Geometry`, `GeometryMesh` exist | `mesh_array`, `plane`, `ground`, `mesh_sequence`, `transformation`, `advanced`, and selection variants are not first-class typed classes yet. |
| Solver | `LinearSolver`, `NonlinearSolver`, `Solver` exist | `line_search`, `augmented_lagrangian`, `rayleigh_damping`, deep linear solver option blocks, and `advanced` are still mostly dict-like. |
| Output | `ParaviewOutput`, `Output` exist | `log`, `paraview.options`, `data`, `reference`, `advanced` are not fully typed classes yet. |
| Time | `Time` exists | Integrator object variants (`BDF`, `ImplicitNewmark`) are still flattened to a string plus a few scalar fields. |
| Boundary conditions | Basic Dirichlet/Neumann exist | `periodic_boundary`, `pressure_boundary`, `pressure_cavity`, `normal_aligned_neumann_boundary`, `obstacle_displacements` still missing as typed classes. |
| Contact | `Contact` exists | `collision_mesh`, `adhesion`, and several booleans/advanced flags from schema are not represented. |
| Selection | `Selection` helper exists | Helpful utility, but not schema-aligned typed selection family. |

### Missing Top-Level Blocks

These schema blocks do not yet have a dedicated public class family.

- `InitialConditions`
- `Constraints`
- `Input`
- `Tests`
- `Space`

## Recommended Next Public Classes

### Priority 1: Geometry + Selection

This is the biggest usability gap for a Python-first API.

Suggested new classes:
- `GeometryPlane`
- `GeometryGround`
- `GeometryMeshArray`
- `GeometryMeshSequence`
- `GeometryTransformation`
- `GeometryAdvanced`
- `BoxSelection`
- `SphereSelection`
- `CylinderSelection`
- `PlaneSelection`
- `AxisSelection`
- `FileSelection`
- `BoxSideSelection`

Useful schema defaults to carry over:
- `geometry[*].type = "mesh"`
- `geometry[*].extract = "volume"`
- `geometry[*].unit = ""`
- `geometry[*].n_refs = 0`
- `geometry[*].enabled = true`
- `geometry[*].is_obstacle = false`
- `geometry[*].transformation.rotation_mode = "xyz"`
- `geometry[*].transformation.translation = []`
- `geometry[*].transformation.rotation = []`
- `geometry[*].transformation.scale = []`
- `geometry[*].transformation.dimensions = 1`
- `geometry[*].advanced.normalize_mesh = false`
- `geometry[*].advanced.force_linear_geometry = false`
- `geometry[*].advanced.refinement_location = 0.5`
- `geometry[*].advanced.min_component = -1`
- selection `id_offset = 0`
- box-side selection requires `threshold`

Why this matters:
- it removes a large amount of raw JSON/dict construction
- it matches the real schema much better than the current `GeometryMesh`-only path
- it would make IDE autocomplete dramatically more useful

### Priority 2: Output Family

Suggested new classes:
- `OutputLog`
- `OutputParaviewOptions`
- `OutputAdvanced`
- `OutputData`
- `OutputDataAdvanced`
- `OutputReference`

Useful schema defaults to carry over:

`output`
- `directory = ""`
- `stats = false`

`output.log`
- `level = "debug"` (string form)
- `file_level = "trace"` (string form)
- `path = ""`
- `quiet = false`

`output.paraview`
- `file_name = ""`
- `vismesh_rel_area = 1e-5`
- `skip_frame = 1`
- `high_order_mesh = true`
- `volume = true`
- `surface = false`
- `wireframe = false`
- `points = false`
- `fields = []`

`output.paraview.options`
- `use_hdf5 = false`
- `material = false`
- `body_ids = false`
- `contact_forces = false`
- `friction_forces = false`
- `normal_adhesion_forces = false`
- `tangential_adhesion_forces = false`
- `velocity = false`
- `acceleration = false`
- `scalar_values = true`
- `tensor_values = true`
- `discretization_order = true`
- `nodes = true`
- `forces = false`
- `force_high_order = false`
- `jacobian_validity = false`

`output.advanced`
- `timestep_prefix = "step_"`
- `sol_on_grid = -1`
- `compute_error = true`
- `sol_at_node = -1`
- `vis_boundary_only = false`
- `curved_mesh_size = false`
- `save_solve_sequence_debug = false`
- `save_ccd_debug_meshes = false`
- `save_time_sequence = true`
- `save_nl_solve_sequence = false`
- `spectrum = false`

Keep the existing Python-only conveniences:
- `save_paraview`
- `save_vtu`
- `result`
- `fallback`

Those should remain hand-designed public API features even though they are not
part of the native PolyFEM JSON schema.

### Priority 3: Solver Family

Suggested new classes:
- `SolverAdvanced`
- `AugmentedLagrangian`
- `RayleighDamping`
- `LineSearch`
- `ArmijoLineSearch`
- `RobustArmijoLineSearch`
- `CollisionSolverOptions` or `SolverContact`
- optionally later: `AMGCLSolver`, `HypreSolver`, `PardisoSolver`

Useful schema defaults to carry over:

`solver`
- `max_threads = 0`

`solver.advanced`
- `check_inversion = "Discrete"`
- `jacobian_threshold = 0`
- `characteristic_length = -1`
- `characteristic_force_density = 10000`
- `cache_size = 900000`
- `lump_mass_matrix = false`
- `lagged_regularization_weight = 0`
- `lagged_regularization_iterations = 1`

`solver.augmented_lagrangian`
- `initial_weight = 1e6`
- `error = 1e-2`
- `scaling = 2.0`

`solver.rayleigh_damping[*]`
- `lagging_iterations = 1`

Time-related nonlinear defaults are already partly represented in the existing
`NonlinearSolver`, but the schema still contains much richer structure than the
current class surface exposes.

### Priority 4: Boundary Conditions + Contact

Suggested new classes:
- `PeriodicBoundary`
- `PressureBoundary`
- `PressureCavity`
- `NormalAlignedNeumannBoundary`
- `ObstacleDisplacement`
- `CollisionMesh`
- `Adhesion`

Useful schema defaults to carry over:

`boundary_conditions.periodic_boundary`
- `enabled = false`
- `tolerance = 1e-5`
- `correspondence = []`
- `fixed_macro_strain = []`
- `linear_displacement_offset = []`
- `force_zero_mean = false`

`contact`
- `enabled = false`
- `dhat = 0.001`
- `dhat_percentage = 0.8`
- `epsv = 0.001`
- `friction_coefficient = 0`
- `use_convergent_formulation = false`
- `use_area_weighting = true`
- `use_improved_max_operator = true`
- `use_physical_barrier = true`
- `use_gcp_formulation = false`
- `alpha_n = 0.5`
- `alpha_t = 0.5`
- `min_distance_ratio = 0.5`
- `use_adaptive_dhat = false`
- `periodic = false`

`contact.collision_mesh`
- `tessellation_type = "regular"`
- `enabled = true`

`contact.adhesion`
- `adhesion_enabled = false`
- `dhat_p = 0.001`
- `dhat_a = 0.01`
- `adhesion_strength = 0.001`
- `tangential_adhesion_coefficient = 0.0`
- `epsa = 0.001`

### Priority 5: Missing Top-Level Blocks

Suggested new classes:
- `InitialConditions`
- `InitialConditionEntry`
- `Constraints`
- `SoftConstraint`
- `Input`
- `Tests`
- `Space`
- `SpaceAdvanced`
- `Remesh`
- `RemeshSplit`
- `RemeshCollapse`
- `RemeshSwap`
- `RemeshSmooth`
- `RemeshLocalRelaxation`

Useful schema defaults to carry over:

`initial_conditions`
- `solution = []`
- `velocity = []`
- `acceleration = []`

`constraints`
- `hard = []`
- `soft = []`
- `soft[*].weight = 0`
- `soft[*].data = ""`

`tests`
- `margin = 1e-5`
- `time_steps = 1`

`space`
- this block is broad; start with `SpaceAdvanced` and `Remesh` before trying to
  encode the entire FE/discretization surface in one pass

## Keep Hand-Written, Do Not Copy Mechanically

Do not expose these generated patterns directly to end users:
- `Object1`, `Object2`, ...
- `Value2`, `Value3`, ...
- `__2` suffixes from duplicated schema variants

These are useful only as internal clues that:
- a path is polymorphic
- the schema has multiple valid forms
- a human-friendly API still needs naming/design work

## Practical Next Iteration

If continuing class development, the most effective next pass is:

1. Expand `Geometry` into a real family of geometry + transformation classes.
2. Replace `Output.log`, `Output.advanced`, and `ParaviewOutput.options` dicts
   with typed classes.
3. Add `SolverAdvanced`, `AugmentedLagrangian`, and `LineSearch`.
4. Add `PeriodicBoundary` and `CollisionMesh` / `Adhesion`.

That sequence gives the largest improvement in Python-first usability while
still keeping the implementation scope manageable.
