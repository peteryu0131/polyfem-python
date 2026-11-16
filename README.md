# PolyFEM for Python

Build a finite element model with `import polyfem` and run it with `from polyfempy import solve`. The Python API configures the simulation and the compiled PolyFEM backend computes the solution.

## Installation

See the [installation guide](INSTALL.md) for pip, conda, and platform status.
The public package indexes still carry older releases; use a reviewed forward
build for the API below. For a development install, see [BUILD.md](BUILD.md).

## Run a forward simulation

Simulate a 1 m x 0.2 m elastic beam. Fix its left end, apply a downward body
force, and compute its displacement. This example uses meters and pascals.

Download [beam.obj](https://raw.githubusercontent.com/polyfem/python_data/6b4751b9c3f40fb9653601efe32ce6b6d0ead96f/quickstart/beam.obj)
(12 KB) and save the following code as `beam.py` in the same folder.

```python
import polyfem
from polyfempy import solve

model = polyfem.model()
beam = model.mesh(mesh="beam.obj")
beam.material(polyfem.linear_elasticity(E=1e7, nu=0.3))

# Fix the left end at x = 0.
left = beam.surface_axis(axis="-x", position=0.0)
left.dirichlet(value=[0, 0])

# Apply a downward body force (N/m^3).
model.rhs([0, -1000])
model.solver(linear=polyfem.linear(solver="Eigen::SimplicialLDLT"))

result = solve(cfg=model.config())
print(f"Largest displacement component: {abs(result.sol).max():.6f} m")
```

Run it from that folder:

```sh
python beam.py
```

The last line should be approximately:

```text
Largest displacement component: 0.003648 m
```

Try changing `E` to adjust the material stiffness, or change the load in
`model.rhs(...)`. The [complete script](https://github.com/polyfem/python_data/blob/6b4751b9c3f40fb9653601efe32ce6b6d0ead96f/quickstart/beam.py)
also works when run from another directory, using the mesh beside it.

`result.sol` is the raw finite-element solution; do not assume that each row
corresponds to a mesh vertex. For a small check that creates its own mesh,
run the [installation self-check](INSTALL.md#check-the-installation).

For larger models, see the [classic example guide](examples/README.md) and the
[pinned example sources](https://github.com/polyfem/python_data/tree/2e533c8357d9f8f0cca4e5baeba391575b2d2f2b/classic_example).
These reference examples require their mesh data and may select linear solvers
that are not enabled in every backend build.

For two more short examples that use only the installed package, see
[static elasticity](examples/quickstart/static_elasticity.py) and
[transient elasticity](examples/quickstart/transient_elasticity.py).
Both create their own mesh and use `Eigen::SimplicialLDLT`. The transient example
also shows time settings and an output directory.

## Source layout

- `python/polyfem/` exposes the model builder and generated configuration helpers.
- `polyfempy/runtime/` implements `solve` and returns the solution in `result.sol`.
- `src/` binds the Python runtime to the C++ solver in the `polyfem/` submodule.
- `python-from-jse/` generates the configuration API from PolyFEM schemas and the rules in `generator-config/`.
- `polyfem-data/` supplies the official meshes, input configurations, and reference values.

## Forward solver validation

On a ready-for-review pull request targeting `main`, GitHub Actions builds the C++ backend, checks a Neo-Hookean cube against reference values, and runs the 75 active contact cases from the pinned PolyFEM test lists. Open the **Generated Contact Backend** run and download **generated-contact-backend-results** for the per-case results. See [testing](TESTING.md) for the commands and triggers.
