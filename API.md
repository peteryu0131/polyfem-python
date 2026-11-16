# Forward API

Start with the runnable example in [README.md](README.md). The snippets below use mesh paths that you supply.

```python
import polyfem
from polyfempy import solve

model = polyfem.model()
body = model.mesh(mesh="part.msh")
body.material(polyfem.neo_hookean(E=1000, nu=0.3))
body.dirichlet(value=[0, 0])

config = model.config(rhs=[0, -1])
result = solve(cfg=config)
```

Use vectors with two components for 2D and three for 3D. Choose values consistent with the model's units.

## Geometry

| Method | Purpose |
| --- | --- |
| `model.mesh(mesh=...)` | Add a mesh |
| `model.mesh_array(mesh=..., array=...)` | Repeat a mesh in an array |
| `model.mesh_sequence(mesh_sequence=..., fps=...)` | Add a sequence of meshes |

Set `is_obstacle=True` for an obstacle. Geometry methods use the names and parameters generated from the spec.

```python
obstacles = model.mesh_array(
    mesh="edge.obj",
    array=polyfem.array(offset=1, size=[21, 1]),
    is_obstacle=True,
)
obstacles.dirichlet(value=[0, 0])
```

The former `obstacle_mesh*` and `geometry_mesh*` methods have been replaced by these geometry methods.

## Materials and body settings

```python
body.material(polyfem.neo_hookean(E=1000, nu=0.3, rho=1000))
body.velocity(value=[0, 0])
body.discr_order(order=1)
```

The returned body connects these settings to its mesh. The builder writes the volume ID when a body-level setting needs it. Creating a mesh alone preserves its existing volume tags.

## All or part of a boundary

Apply a condition directly to the body to select its entire outer boundary.

```python
body.dirichlet(value=[0, 0])
```

For a partial boundary, select it first and then apply the condition.

```python
# Select the left side of the mesh bounding box.
boundary = body.surface_axis(axis="-x", position=0.0)
boundary.dirichlet(value=[0, 0])
```

The selection parameters follow the PolyFEM spec. The API assigns the boundary ID and connects the condition to it. The same binding mechanism supports `neumann`, `normal_aligned_neumann` and `pressure`.

A direct whole-boundary condition and a partial or file-based surface selection on the same body are rejected when they would overwrite each other's selection. Apply conditions to explicit selections when a body has multiple boundary regions.

For an existing sideset file:

```python
boundary = body.surface_from_file("surface_sidesets.txt")
boundary.dirichlet(value=[0, 0])
```

If the file contains several groups, supply `tag=...` to identify the intended group.

Use `surface_all()` when you need an explicit whole-boundary object. It also supports shared labels:

```python
first_surface = first.surface_all()
second_surface = second.surface_all(same_as=first_surface)
first_surface.dirichlet(value=[0, 0])
```

Here both surfaces share one backend label, so the condition applies to both.

## Body loads

For a uniform load across the model:

```python
config = model.config(rhs=[0, -1])
```

For different loads on different bodies:

```python
first.rhs(value=[0, -1])
second.rhs(value=[0, 0])
config = model.config()
```

Choose one of these load scopes per model. Supplying both a global RHS and body-specific RHS raises an error.

## Time, contact and output

```python
config = model.config(
    time=polyfem.time(dt=0.01, tend=0.5, integrator="BDF"),
    contact=polyfem.contact(enabled=True, dhat=0.001),
    output=polyfem.output(
        directory="results/example",
        json="sim.json",
        paraview=polyfem.output_paraview(file_name="sim.pvd"),
    ),
)
result = solve(cfg=config)
```

A relative output directory is resolved from the working directory. `result.sol` holds the solver's solution array. Classic examples explicitly retain their source JSON settings, including defaults.

Configuration sections can also be set with `model.time(...)`, `model.output(...)` and other generated section methods. Use one configuration style for each section. Passing a section to `model.config(...)` after setting it on the model raises an error.

## Generation and maintenance

Configuration factories and geometry entry points are generated from the pinned specs. Shared builder code manages scene objects and IDs. `generator-config/id_relationships.json` defines how bodies and selections connect to materials, loads and boundary conditions.

PolySolve specs are cached by the revision selected by the pinned PolyFEM source. After updating the source or generator, regenerate and test using [BUILD.md](BUILD.md) and [TESTING.md](TESTING.md).
