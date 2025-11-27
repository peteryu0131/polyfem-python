# Legacy API

This directory contains the original PolyFEM Python API bindings.

## What's Here

- `Problem.py`: Generic problem base class
- `Problems.py`: Predefined problems (Franke, Gravity, Torsion, etc.)
- `Selection.py`: Geometry selection utilities
- `autoclass.py`: Auto-generated dataclass utilities
- `command.py`: Command-line interface for PolyFEM

## Why Moved?

These files were moved here from the root `polyfempy/` directory to:
1. **Separate new from old**: The new simplified API lives in `polyfempy.api`
2. **Preserve backward compatibility**: Keep advanced/C++ features available
3. **Reduce confusion**: Clear separation between modern and legacy code
4. **Clean organization**: All legacy code is now consolidated in one place

**Note**: These files were previously in the root `polyfempy/` directory but have been moved to `polyfempy/legacy/` to avoid confusion. The new API uses different implementations (e.g., `polyfempy.api.selection.Selection` instead of `polyfempy.legacy.Selection`).

## Usage

### Legacy API (Advanced)

```python
from polyfempy.legacy import Problem, Problems, Selection
from polyfempy import Settings, Solver

# Use the old C++ binding style
problem = Problems.GenericTensor()
problem.add_dirichlet_value(1, [0, 0])
settings = Settings()
settings.set_problem(problem)
# ...
```

### New API (Recommended)

```python
from polyfempy.api import solve, SimulationConfig

cfg = SimulationConfig()
result = solve(V, C, cfg, backend="dummy")
```

## Migration

If you have existing code using the legacy API, you can:
1. Keep using it: `from polyfempy.legacy import ...`
2. Migrate to new API: See `polyfempy/api/` for simplified interface
3. Use both: They can coexist


