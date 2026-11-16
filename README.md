# PolyFEM for Python

Build a finite element model with `import polyfem` and run it with `from polyfempy import solve`. The Python API configures the simulation and the compiled PolyFEM backend computes the solution.

## Installation

See the [installation guide](INSTALL.md) for pip, conda, and platform status.
The public package indexes still carry older releases; use a reviewed forward
build for the API below. For a development install, see [BUILD.md](BUILD.md).

## Run a forward simulation

After installation, run this example to create a square mesh with four triangles,
fix its boundary, and apply a uniform body load. Only the center vertex can move.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import polyfem
from polyfempy import solve

with TemporaryDirectory() as folder:
    mesh = Path(folder) / "square.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv .5 .5 0\n"
        "f 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n"
    )

    model = polyfem.model()
    square = model.mesh(mesh=str(mesh))
    square.material(polyfem.neo_hookean(E=1000, nu=0.3))
    square.surface_all().dirichlet(value=[0, 0])
    model.rhs([0, -1])

    result = solve(cfg=model.config())
    print(abs(result.sol).max())
```

The largest absolute displacement component is about `0.00011235`.
`result.sol` is the raw finite-element solution; do not assume that each row
corresponds to a mesh vertex. The example creates its own mesh and does not need
the data or examples repositories at runtime once the package is installed.

For larger models, see the [classic example guide](examples/README.md) and the
[pinned example sources](https://github.com/polyfem/python_data/tree/2e533c8357d9f8f0cca4e5baeba391575b2d2f2b/classic_example).
These reference examples require their mesh data and may select linear solvers
that are not enabled in every backend build.

For two short examples that use only the installed package, start with
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
