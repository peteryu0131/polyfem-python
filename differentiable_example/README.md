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
