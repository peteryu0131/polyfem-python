# Objective configuration and support

## Standard configuration

`diff.objectives.create(type_name, **fields)` uses the names, fields, and
defaults extracted from `polyfem/json-specs/objective-spec.json`. The snapshot
covers all 41 standard types. `diff.objectives.standard_types` lists them.
For objectives requiring a state, this Python entry supplies `state=0` when
omitted; this is a single-state convenience, not a schema default.

```python
objective = diff.objectives.create("compliance", volume_selection=[2])
final_objective = diff.objectives.transient_integral(
    static_objective=objective, integral_type="final",
)
```

New convenience factories replace hyphens with underscores, for example
`displacement_target` serializes as `displacement-target`. Existing helpers
such as `compliance(selection=body)` retain their signatures. Use `create`
when passing exact schema field names to those existing helpers.

Volume selections accept ID lists or file paths; surface selections accept
an ID or ID list. Body handles can populate volume selections; boundary
handles can populate surface selections. The `selection` alias is available
only when the objective has one unambiguous selection domain.

Construction checks required fields, unknown fields, basic types, enum
options, bounds, and recursively nested objectives. Opaque parametrizations,
mesh-dependent dimensions, and backend execution support are not certified
by these configuration checks. A valid configuration does **not** imply that
its value or gradient is supported by this binding. In particular, exposing
`max_stress` does not implement its missing derivative.

## Local extensions

`diff.objectives.extension_types` lists `von_mises_peak` and
`von_mises_squared`. Use their explicit helpers, not the standard `create`
entry. They are not aliases for standard `max_stress`. Standard recursive
composition accepts standard objectives only; the existing
`von_mises_squared(...).final()` extension remains supported separately.

## PyTorch optimization

The optimizer loop remains `zero_grad(); loss = problem(); loss.backward();
optimizer.step()`. Objective configuration does not select or replace the
PyTorch optimizer.

The current binding already uses these variable mappings from
`opt-input-spec.json`:

| Python primitive | Native mapping type | Parameters |
|---|---|---|
| `shape_opt` | `shape` | Mesh vertices, optionally mapped from user parameters |
| `material_opt` | `elastic` | Elastic parameters, including the existing E/nu conversion |
| `initial_condition_opt` | `initial` | Initial displacement and velocity |

This work does not add the native optimization driver or all mappings from
that schema. The existing material and initial-condition paths use one state;
accepting a schema state index does not add multi-state execution support.

Regenerate the metadata after updating the upstream schema:

```powershell
python tools/generate_objective_schema.py
python tools/generate_objective_schema.py --check
```

The snapshot is packaged with Python, so installed users do not need the
source JSON files. Configuration tests check snapshot drift and every standard
type without running FEM.

## Stress objective support

This table describes the local binding, not every upstream PolyFEM feature.
JSON schemas list configurable names; they do not certify gradient support.

| Objective | Value | Material gradient | Time scope |
|---|---|---|---|
| `von_mises_peak` | Cauchy von Mises maximum at element quadrature points | Implemented, static LinearElasticity/NeoHookean; transient NeoHookean, 2D/3D | All cached steps, including initial state; static single state also supported |
| `von_mises_squared` | Reference-volume integral of squared Cauchy von Mises | Implemented for the same materials | Static or `.final()` only |
| `max_stress` | Maximum PK1 tensor Frobenius norm | Not implemented | Spatial objective; no whole-time maximum wrapper |
| `stress_norm` | Integral of a stress tensor norm power | Not implemented | Spatial objective / existing time integration wrappers |

`von_mises_peak` also supports shape gradients for P1 simplex meshes with
uniform numeric material properties, using the existing transient contact
adjoint and the pointwise objective's direct geometry derivative.
Shape optimization with contact currently requires a fixed numeric
`solver.contact.barrier_stiffness`. Adaptive stiffness is rejected before a
solve: caching each step's stiffness fixes historical force evaluation, but
does not differentiate the adaptive stiffness selection/update algorithm.
The adaptive small-model finite-difference check failed; do not treat the
fixed-stiffness checks as validation of adaptive-contact shape gradients.
`von_mises_squared` still rejects shape optimization. Both reject initial-condition optimization.
An empty selection includes every body; state must be zero. In 2D they use the
backend xx^2 - xx*yy + yy^2 + 3*xy^2 convention, without inferred zz stress.
Peak units are internal stress times weight, with no volume/time integration.
No new dependencies or generated configuration schema changes are required.

`problem = diff.material_opt(...)`; after `problem()`, `problem.diagnostics`
contains peak value, step, physical time, zero-based element/quadrature indices,
and reference position. It is replaced on each evaluation; copy it to retain it.

The exact peak is nonsmooth when maximizers switch. Exact ties use the first
step/element/quadrature sample; the zero norm uses a zero subgradient. This is a
branch/subgradient convention, not a guarantee of differentiability at ties.
The dynamic adjoint propagates sensitivity through the cached trajectory.
Numerical tests on small meshes do not establish gradient accuracy for arbitrary
adaptive-contact simulations. Time-step/mesh/solver accuracy remains necessary.

The new peak objective rejects transient LinearElasticity before solving: the
upstream linear transient solver does not initialize the nonlinear problem
required by the dynamic differentiation cache. This limitation is not repaired
by the peak objective. NeoHookean is the tested transient route here.
