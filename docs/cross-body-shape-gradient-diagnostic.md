# Cross-Body Shape Gradient Diagnostic

This note records the current status of experiment 02 cross-body shape
gradients.

## Supported Path Today

The current Python-exposed differentiable path works for the same body objective
and design body:

```text
loss = lattice von Mises objective
design variable = lattice shape vertices
gradient = d(lattice VM) / d(lattice vertices)
```

In the experiment 02 scripts this is:

```python
make_von_mises_loss(
    result=result,
    volume_selection=1,          # lattice
    time_aggregation="smooth_max",
)
loss.backward()
```

The resulting `result.shape_gradient` is nonzero on lattice vertices.

## Desired Cross-Body Path

The harder target is:

```text
loss = block von Mises objective
design variable = lattice shape vertices
gradient = d(block VM) / d(lattice vertices)
```

This is a cross-body/contact sensitivity. The forward simulation can make the
block stress depend on the lattice geometry, but the adjoint path also has to
carry that sensitivity through contact back to the lattice vertices.

## Diagnostic Script

Run:

```bash
source "$HOME/polyfem_env/bin/activate"
python -m experiment.experiment_api_solve.diagnose_experiment02_cross_body_shape_gradient
```

The script compares:

```text
finite difference:
    [J_block(V_lattice + eps * direction) - J_block(V_lattice - eps * direction)] / (2 eps)

current adjoint:
    dJ_block/dV, masked to lattice vertices
```

It writes both JSON and text reports into the run workspace:

```text
cross_body_shape_gradient_probe.json
cross_body_shape_gradient_probe.txt
```

## How To Interpret

If the report shows:

```text
finite_difference_directional_derivative != 0
adjoint_directional_derivative = 0
probe_body_grad_norm = 0
```

then the forward simulation has cross-body sensitivity, but the current
Python-exposed adjoint path does not return
`d(block VM) / d(lattice vertices)`.

This does not prove that PolyFEM has no contact shape derivative machinery.
It means the currently exposed pipeline does not connect or return this
cross-body objective-to-design gradient.

## Where The Missing Work Likely Is

Python can express:

```python
objective_body_id = 2  # block
design_body_id = 1     # lattice
```

but Python cannot create a missing adjoint term. The C++ adjoint chain must
provide the full derivative:

```text
block VM objective
  -> dJ/du
  -> transient adjoint solve
  -> contact force shape derivative dF_contact/dX_lattice
  -> gradient on lattice vertices
```

The local binding currently exposes a generic:

```cpp
shape_derivative(solver)
```

and objective partial derivatives via:

```cpp
Objective.derivative(..., wrt="solution")
Objective.derivative(..., wrt="shape")
```

There is no high-level exposed API that explicitly separates:

```text
objective body = block
design body = lattice
```

So if finite difference is nonzero while the lattice adjoint gradient is zero,
the next implementation target is the C++ transient shape-adjoint/contact
derivative path, not only the Python wrapper.

## Short Message For Discussion

```text
I tested block von Mises with respect to lattice shape using finite difference.
The finite-difference directional derivative on lattice vertices is nonzero,
but the current adjoint gradient masked to lattice vertices is zero.

So the forward simulation has cross-body sensitivity, but the current
Python-exposed adjoint path does not return d(block VM)/d(lattice vertices).
The same-body path, d(lattice VM)/d(lattice vertices), works and is still valid
for the current training/optimization example.
```
