# Differentiable Python Porting Notes

This note is for porting the current Python-side differentiable fixes to an
older checkout. Most items are pure Python changes. Rebuilding the C++ extension
is only required if the older binding lacks `solve(log_level=...)` or
`get_solutions()`.

The fixes address two common symptoms:

1. terminal logs do not match the configured `polyfem.log` verbosity
2. transient `loss.backward()` fails with `Invalid adjoint_rhs shape!`

The current implementation lives mainly in:

```text
polyfempy/differentiable/solve_diff.py
polyfempy/differentiable/torch_integration.py
```

## `solve_diff.py`

### Console Log Level

Add or keep `_console_log_level_from_settings(settings)`. It should read
`settings["output"]["log"]["level"]` and map PolyFEM log names to the integer
levels passed to `solver.solve(log_level=...)`.

### Quiet Setup

`solve_differentiable(...)` should accept:

```python
quiet_polyfem_setup: bool = True
```

When this flag is true, setup-time PolyFEM output can be muted. When false,
setup should preserve the configured console log level so terminal output and
`polyfem.log` are easier to compare.

### Pass Solve Log Level Into The Autograd Function

After converting config to settings:

```python
solve_log_level = _console_log_level_from_settings(settings)
solutions = PolyFEMFunction.apply(solver, V_torch, derivative_type, solve_log_level)
```

Older code that called `PolyFEMFunction.apply(solver, V_torch, derivative_type)`
needs the matching `forward(...)` signature update described below.

### Do Not Pre-Run Transient Setup For Disk Meshes

For the disk-mesh path, the setup order should stop at loading mesh data and
building the basis before entering `PolyFEMFunction.forward`. Do not pre-run:

- `assemble()`
- `set_cache_level(Derivatives)`
- `init_timestepping(...)`

The forward pass owns the unique execution order:

```text
set_vertices
  -> build_basis
  -> assemble
  -> set_cache_level
  -> solve(log_level=...)
```

Running transient setup twice can leave solver state inconsistent, especially
for larger Neo-Hookean transient cases.

## `torch_integration.py`

### Forward Signature

`PolyFEMFunction.forward` should accept:

```python
solve_log_level: int = 2
```

Call C++ as:

```python
try:
    ret = solver.solve(log_level=solve_log_level)
except TypeError:
    ret = solver.solve()
```

The fallback keeps very old bindings usable, but the preferred binding accepts
`log_level`.

### Prefer `get_solutions()` For Transient Displacements

After `solver.solve(...)`, build `solutions_np` in this order:

1. `np.asarray(solver.get_solutions())`, if available and non-empty
2. the returned bundle, such as `ret["u"]` or tuple output
3. any final compatibility fallback

The adjoint path expects the displacement tensor layout to match the internal
differentiable cache. For transient solves, `get_solutions()` is the safest
source.

### Backward Return Arity

If `forward(...)` has four inputs, `backward(...)` must return four entries:

```python
return None, grad_tensor, None, None
#      solver  vertices     derivative_type  solve_log_level
```

Non-tensor inputs get `None` gradients.

## Application Layer

For scripts that call `solve_differentiable`, expose a CLI flag or config option
when console logs are needed:

```python
quiet_polyfem_setup=False
```

The default can remain true for cleaner output.

## Validation Checklist

- [ ] `solve_differentiable` passes `solve_log_level` into `PolyFEMFunction.apply`.
- [ ] `PolyFEMFunction.forward` calls `solver.solve(log_level=solve_log_level)` when supported.
- [ ] transient displacement extraction prefers `solver.get_solutions()`.
- [ ] `backward` returns one slot per `forward` input.
- [ ] a transient differentiable case can run `loss.backward()` without an adjoint RHS shape error.

## If Porting Still Fails

- Confirm Python imports the checkout you edited.
- If `solver.solve(log_level=...)` always raises `TypeError`, update the C++
  binding in `src/state/state.cpp` and rebuild.
- If `get_solutions()` is missing, update the binding or backport the method;
  hand-constructing an equivalent transient displacement layout is not
  recommended.
