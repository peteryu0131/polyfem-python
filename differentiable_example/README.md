# Differentiable examples

Run from the repository root using the environment containing PyTorch and the
compiled PolyFEM backend:

```powershell
conda run -n polyfem python -m polyfempy backend-info
conda run -n polyfem python differentiable_example/initial_condition/projectile_angle.py
```

Initialize `polyfem-data` and `differentiability-data` submodules if their meshes
are missing. Metadata helpers require a backend built from the current source.
The shape examples also require the Pardiso solver used by the reference case.

Thread limits use the same configuration as ordinary forward solves:
`pf.solver(max_threads=4)` requests a maximum of four threads; omitting the
setting uses the backend default `0` (no additional limit). Use an explicit
`max_threads=1` for single-thread comparisons. Older differentiable binaries
silently forced this setting to one; rebuild after updating the binding.
`DifferentiableSession.settings()` returns resolved backend settings as JSON
for inspection without solving. A thread limit is not a guarantee that every
operation uses that many threads or runs proportionally faster.

## Start here

| Example | What to learn |
| --- | --- |
| [Launch angle](initial_condition/projectile_angle.py) | Angle → velocity → simulation → range loss → PyTorch optimizer |
| [Material compliance](material/linear_elasticity_compliance.py) | Initialize per-element Lamé parameters and perform two demonstration optimization updates |
| [Shape gradient](shape/neohookean_stress_3d_opt.py) | Compute one stress objective and its gradient with respect to mesh vertices |

The launch example reconstructs equal-height landing under constant gravity;
it does not simulate ground contact. Its position integral equals the mean only
for the centered unit-area mesh used here. Changing geometry may require adapting
the loss, even when the metadata correctly updates tensor sizes.

## Further examples

### Von Mises material gradient (validation milestones)

`diff.objectives.von_mises_squared(selection=body)` integrates squared **Cauchy**
von Mises stress over the body's reference volume. It includes the direct
material derivative and the displacement contribution through the adjoint.
It supports static LinearElasticity / NeoHookean material differentiation and
provides a final-time path for transient models via `.final()`. Other time
aggregations and shape/initial-condition differentiation remain unsupported.

For an existing static model with two bodies, prepare material tensors in actual
backend element order. Query metadata once and rebuild derived tensors on each
loss evaluation:

```python
metadata = diff_model.metadata()
E = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
fixed_lame = torch.tensor([60.0, 60.0], dtype=torch.float64)

def loss():
    nu = 0.25
    pair = torch.stack([E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))])
    lame = metadata.material_by_body({body: pair, other_body: fixed_lame})
    return diff.material_opt(diff_model, lame,
                            objective=diff.objectives.von_mises_squared(selection=body))()

value = loss()
value.backward()  # E.grad includes E -> Lame -> simulation -> stress objective.
```

All material tensors above use **backend internal units**, not necessarily the
input JSON's stress units. The objective has stress-squared times reference-volume
units. In 2D it integrates reference area with unit thickness and uses the backend
convention `sxx² - sxx*syy + syy² + 3*sxy²` (no inferred out-of-plane stress).
It is not peak stress. `material_by_body` requires exactly one pair per mesh body;
body handles or integer IDs work, including ID 0 for a raw mesh.

After rebuilding, run the dedicated small checks from the repository root:

```powershell
conda run -n polyfem cmake --build build/temp.win-amd64-cpython-311/Release --config Release --target polyfempy --parallel 1
conda run --no-capture-output -n polyfem python -m pytest tests/test_differentiable_api_von_mises.py -v
```

The checks use two four-element bodies in 2D/3D, compare prescribed-strain values
with analytic values, compare gradients against central differences at two step
sizes, and exercise body selection and unsupported modes. They disable VTU output
and do not run the lattice impact experiment.

For the separate, small cross-body contact check, use
`diff.objectives.von_mises_squared(selection=upper_body).final()` while only the
lower body's E is trainable. The test creates two 8-triangle rectangles and uses
20 BDF1 steps (`dt=0.004`, `T=0.08`). The lower bottom is fixed; the upper block
starts with uniform downward velocity, with zero body forces. Fixed barrier
stiffness and zero friction isolate the basic coupling. The objective is the
upper block's final stress-squared reference-area integral, not its time peak.

```powershell
conda run --no-capture-output -n polyfem python -m pytest tests/test_differentiable_api_contact_material.py tests/test_differentiable_api_von_mises.py -v -s -x
```

The contact case requires a nonzero cross-body gradient agreeing with central
differences at two relative E perturbations (0.001 and 0.0003), within 1% relative
and 1e-10 absolute tolerance. Separated and contact-disabled controls must have
negligible upper stress and cross-body gradient. No VTU output or full lattice
simulation is used. The new transient path and fixture still require local
compilation and execution; these settings are not claimed to be validated until
those checks pass. Passing them would not validate adaptive contact stiffness,
full-lattice peak optimization or physical accuracy.

### One bounded update with independent forward verification

The contact fixture uses the backend's **2D** E/nu conversion:
`lambda = E*nu/(1-nu**2)`, `mu = E/(2*(1+nu))`. This differs from the 3D lambda
conversion. The earlier contact fixture used the 3D mapping; its finite-difference
check validated that mapping, but did not establish equivalence with a 2D JSON
E/nu material. The corrected fixture must be checked again.

`test_one_lower_E_update_reduces_upper_objective_in_independent_forward` uses the
existing compact optimizer for one Adam update (`lr=0.1`, lower E initially 100,
bounds 90–110). The upper block's E remains 500 and nu remains 0.25 for both.
These are small-model internal units, not the paper experiment's MPa range.
The recorded optimizer loss is pre-update, so the test explicitly re-evaluates
the objective after the update.

Two separate ordinary `polyfempy.solve` calls then use the initial and updated
lower E in their material configurations. Each saves just frames 0 and 20 in
the pytest temporary directory. The test recomputes the upper block's weighted
reference-area stress-squared integral from raw (not averaged) VTU stress values.
P1 affine triangles have constant stress per element; the test checks this and
the total reference area before integrating. Both values must match the
differentiable path, and the updated value must decrease. It prints and saves
`optimization_check.json` with the two E values, gradient, objective values and
relative reduction. No C++ rebuild is needed for this Python-only test change.

Run the corrected gradient check and the new optimization check (nine tiny
forward solves and two backward calls in total), stopping at the first failure:

```powershell
conda run --no-capture-output -n polyfem python -m pytest tests/test_differentiable_api_contact_material.py::test_contact_transmits_lower_E_gradient_to_upper_stress tests/test_differentiable_api_contact_material.py::test_one_lower_E_update_reduces_upper_objective_in_independent_forward -v -s -x
```

This tests one local update, not convergence to an optimal E or a reduction in
mean/peak stress. The new check has not yet been executed by the authoring agent.

### Other examples

| Example | Distinct purpose |
| --- | --- |
| [Fixed-time position](initial_condition/transient_elastic_initial_condition.py) | Velocity and angle gradients against a fixed-time target, without gravity |
| [Shape optimization](shape/neohookean_stress_3d_optimization.py) | Update vertices with the compact API; an equivalent PyTorch loop is included as comments |
| [Shape chain rule](shape/neohookean_stress_3d_chain_rule.py) | Differentiate with respect to scalar x-scale and compose a squared stress loss |

Optimization iterations count parameter updates; `dt` and `tend` control time
integration inside a simulation. Each script writes results under its `runs/`
directory or prints a JSON summary.

The material and initial-condition examples enable the compact API and retain
the equivalent full PyTorch loop as comments. Choose one block and rerun the
script from the beginning; do not execute both blocks in sequence.

```python
# Direct leaf material parameters (direct initial-condition tensors work too).
result = material_opt.optimizer(torch.optim.Adam, lr=1.0).steps(2).optimize()

# A derived parameter: rebuild theta -> velocity -> simulation on every call.
result = (
    diff.optimization(lambda: loss(theta), parameters=[theta])
    .optimizer(torch.optim.Adam, lr=0.05)
    .steps(8)
    .optimize()
)
```

The shared loop supports SGD/Adam-style optimizers, not closure-based LBFGS.
Shape, material, and initial-condition optimization all use this loop and expose
`result.parameters`, `result.history`, and `result.loss`. Shape results retain
the `ShapeOptimizationResult` compatibility type (a subclass of
`OptimizationResult`) and `result.vertices` aliases `result.parameters[0]`.
Existing shape `optimize(optimizer=..., steps=...)` calls remain supported.
Optional `record(step, loss)` returns extra history fields after backward and
before the update. `bounds=[(lower, upper)]` clamps the parameter after each
update, as in the launch example. History and `result.loss` describe pre-update
evaluations; explicitly evaluate the loss again for a final post-update value.
The material example only demonstrates a few updates, not a constrained design
optimum with material budgets or physical bounds.

Use `diff_model.metadata()` once outside the loop, then rebuild differentiable
parameter mappings inside the loss. See [shared parameter preparation](../README.md#shared-parameter-preparation)
for shared versus independent material parameters and supported FE spaces.
`shape/_shape_common.py` only loads the reference Gmsh vertices; it is not a
general mesh-import API.

`paper_example/` now includes a [forward-only lattice impact baseline](paper_example/README.md)
using the new API, with contact and stress-output checks. The original JSON is
retained as reference data; optimization objectives and parameter maps are not
yet migrated.
