# PolyFEM generator configuration

The API generator reads these project settings during generation.

- `builder.json` selects the PolyFEM builder, its extra helpers and the spec paths to exclude.
- `api_aliases.json` defines public names, field aliases and configuration shortcuts.
- `id_relationships.json` connects geometry IDs to materials, boundary conditions and contact settings. Its `id_groups` also defines which selection types share IDs.

The generic generator lives in the `python-from-jse/` submodule.

## Generated global configuration helpers

The generic generator accepts keyword arguments directly on polymorphic
factories, for example `polyfem.time(dt=0.01, tend=0.5, integrator="BDF")`.
It selects a schema branch using its required fields; explicit variant objects
remain supported for compatibility.

Root configuration sections also generate methods on the model builder:

```python
import polyfem

model = polyfem.model()
model.time(dt=0.01, tend=0.5, integrator="BDF")
model.units(length="cm", mass="g", time="s")
model.output(directory="runs/example")
config = model.config()
```

These setters return the model and replace the same section when called again.
Passing that section again to `model.config(...)` raises an error instead of
silently overriding it. Shared builder methods also support setting `space`,
`materials`, and `initial_conditions`, while preserving their no-argument reads.
`boundary_conditions(...)` saves keyword settings but keeps its dictionary return
value for existing callers. Passing that unchanged dictionary to `model.config(...)`
is supported without duplicating boundary entries.

Explicit section settings are combined with body settings when exporting the
model. Lists are appended; conflicting non-list fields raise an error. Repeating
a setter replaces only its explicit settings. ID assignment remains in the
shared builder and relationship rules, so users can work with body and selection
objects. See [Model settings](../API.md#model-settings) for examples and merge rules.

The generator creates methods from the input spec. PolyFEM implements mesh,
obstacle and RHS behavior in `python/polyfempy/_model_builder.py`. A removed
field referenced by a configured shortcut or relationship fails validation
before the generated files are replaced. Regenerate from the repository root.

```powershell
python tools/generate_polyfem_api.py
```

## Geometry and ID binding

Geometry factories also generate model methods such as `model.mesh(...)`,
`model.mesh_array(...)` and `model.mesh_sequence(...)`. Use
`is_obstacle=True` to mark an obstacle.

Shared builder logic registers geometry and binds IDs. Body-level consumers
such as `body.material(...)` use volume relationships. Direct boundary
conditions such as `body.dirichlet(...)` use the configured surface
relationships and select the whole boundary. Partial selections retain their
own handles.

Selection helper names and fields come from the spec and the relationships.
For example, the surface relationships identify `geometry[*].surface_selection`
and its ID, and the selection spec defines the `axis` variant and its arguments.
Together they generate `surface_axis(...)`. Adding a named selection variant
with that ID generates its helper on the next run.

Each relationship namespace gets its own ID group by default. The project
declares that surface and point share a group and must expose matching methods.
It also reserves curve IDs, which have no corresponding consumer relationship.
The builder can customize a root section through a private `_section_<name>`
method. That method is exposed only when the section exists in the spec.
`builder.source_root` points to the package source relative to `builder.json`,
so the generator can read the adapter's declared methods and retain their type
signatures without importing the package during generation.
A collection handler such as `_item_geometry` receives the generated item
factory name. The spec supplies the collection and its variants.

See [API.md](../API.md) for user-facing examples. New geometry types use the
same generated dispatch. New relationships must be described in
`id_relationships.json` and tested against the backend configuration.
