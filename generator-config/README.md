# Generator Config

This directory is the future home for PolyFEM-specific generator configuration.

The generator itself should stay generic and live under `python-from-jse/`.
PolyFEM-specific API decisions belong here, because they describe how the
PolyFEM Python binding wants to use the generic generator.

Active files:

- `api_aliases.json`: public API names, aliases, hidden generated names, and
  user-facing rename rules.
- `id_relationships.json`: builder/model relationships, such as how selections
  connect geometry, materials, boundary conditions, and contact settings.

Possible future file:

- `schema_patches.json`: PolyFEM schema patches or overlays that should be
  applied before generating the schema-faithful classes. Do not add this until
  PolyFEM has a real schema gap that should be handled outside the backend JSON
  specs.

Current state:

- The active PolyFEM config files live in this directory.
- Generator tools and validators should read this directory by default.
- Generic generator examples or dummy config may still live under
  `python-from-jse/` when they document supported generator capabilities.
- The current `schema_patches` examples are generic examples under
  `python-from-jse/examples/`; they are not active PolyFEM config.

## Generated global configuration helpers

The generic generator accepts keyword arguments directly on polymorphic
factories, for example `pf.time(dt=0.01, tend=0.5, integrator="BDF")`.
It selects a schema branch using its required fields; explicit variant objects
remain supported for compatibility.

Root configuration sections also generate methods on the model builder:

```python
model = pf.model()
model.time(dt=0.01, tend=0.5, integrator="BDF")
model.units(length="cm", mass="g", time="s")
model.output(directory="runs/example")
config = model.config()
```

These setters return the model and replace the same section when called again.
Passing that section again to `model.config(...)` raises an error instead of
silently overriding it. Existing builder methods (including `space()` and
body/selection helpers) retain their existing behavior; they are not replaced
by generated setters.

Maintain this behavior in `python-from-jse/generator/JsonToTreeClass.py` and
`python-from-jse/generator/model_builder.py`, not the generated output. From the
repository root, regenerate with:

```powershell
conda run -n polyfem python tools/generate_polyfem_api.py
```
