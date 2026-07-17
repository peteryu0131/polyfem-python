# Generator Config

This directory is the future home for PolyFEM-specific generator configuration.

The generator itself should stay generic and live under `python-from-jse/`.
PolyFEM-specific API decisions belong here, because they describe how the
PolyFEM Python binding wants to use the generic generator.

Planned files:

- `api_aliases.json`: public API names, aliases, hidden generated names, and
  user-facing rename rules.
- `id_relationships.json`: builder/model relationships, such as how selections
  connect geometry, materials, boundary conditions, and contact settings.
- `schema_patches.json`: PolyFEM schema patches or overlays that should be
  applied before generating the schema-faithful classes.

Current state:

- The active PolyFEM config files live in this directory.
- Generator tools and validators should read this directory by default.
- Generic generator examples or dummy config may still live under
  `python-from-jse/` when they document supported generator capabilities.
