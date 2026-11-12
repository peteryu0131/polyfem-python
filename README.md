# PolyFEM for Python

Build a finite element model with `import polyfem` and run it with `from polyfempy import solve`. The Python API configures the simulation and the compiled PolyFEM backend computes the solution.

## Run a forward simulation

Install the bindings from this checkout using the [build instructions](BUILD.md), or install a wheel built from it. This example creates a square mesh with four triangles, fixes its boundary, and applies a uniform body load. Only the center vertex can move.

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

The largest absolute displacement component is about `0.00011235`. For larger models, see the [classic examples](https://github.com/polyfem/python_data/tree/89773e1c485a223b230fe372ca2032122cd0a7cd/classic_example).

## Source layout

- `python/polyfem/` exposes the model builder and generated configuration helpers.
- `polyfempy/runtime/` implements `solve` and returns the solution in `result.sol`.
- `src/` binds the Python runtime to the C++ solver in the `polyfem/` submodule.
- `python-from-jse/` generates the configuration API from PolyFEM schemas and the rules in `generator-config/`.
- `polyfem-data/` supplies the official meshes, input configurations, and reference values.

## Forward solver validation

On a ready-for-review pull request targeting `main`, GitHub Actions builds the C++ backend, checks a Neo-Hookean cube against reference values, and runs the 75 active contact cases from the pinned PolyFEM test lists. Open the **Generated Contact Backend** run and download **generated-contact-backend-results** for the per-case results. See [testing](TESTING.md) for the commands and triggers.
