# PolyFEM Python API Architecture

This document explains all files in the `polyfempy/api/` directory, including design philosophy, function purposes, and examples.

> Note (Route A): Python should always `import polyfempy as pf` (stable C++ extension module name).
> nanobind vs pybind11 is a build-time choice and must not affect imports. The C++ binding’s
> `Solver.solve()` returns `(sol, pressure)`, so Python callers must capture and parse it.

## Directory Structure

```
polyfempy/api/
├── __init__.py              # Module entry, exports main API
├── solve.py                 # Main solve function (unified entry point); calls C++ directly (Route A)
├── config.py                # Configuration (SimulationConfig + material/BC/geometry/solver classes)
├── result.py                # Result container (Result)
├── selection.py             # Geometry selection for boundary conditions (Selection)
├── batch.py                 # Batch processing (batch_solve)
├── tensor.py                # Tensor conversion (multi-backend NumPy/Torch/JAX)
└── io.py                    # Mesh I/O (read_mesh, Mesh; uses meshio)
```

Examples live in the repository root under `examples/` (e.g. `examples/python_config_5_cubes.py`), not inside `polyfempy/api/`.

---

## 1. `__init__.py` - Module Entry

### Design Philosophy

**Single Entry Point**: Only export the main API that users need, hide implementation details.

### Exports

```python
from .solve import solve
from .config import SimulationConfig
from .result import Result
from .selection import Selection
from .batch import batch_solve

__all__ = ["solve", "SimulationConfig", "Result", "Selection", "batch_solve"]
```

### Function Purposes

- **`solve`**: Main solve function, primary entry point for users
- **`SimulationConfig`**: Configuration class for creating simulation configurations
- **`Result`**: Result container containing solution and metadata
- **`Selection`**: Geometry selection tool for selecting boundary conditions by geometric shapes
- **`batch_solve`**: Batch solve (sequential, order-preserving)

### Examples

- Example scripts: `examples/` at the repository root (e.g. `examples/python_config_5_cubes.py`, `examples/contact_5_cubes.py`).

---

## Core Concepts

Before diving into each module, understand these core concepts that run through the entire architecture:

### Multi-Backend Array Support (NumPy/Torch/JAX)

**Design Reasons**:
- Users may use different array libraries (NumPy, PyTorch, JAX)
- Support PyTorch and JAX automatic differentiation
- Provide unified API, hide backend differences

**Implementation**:
- `tensor.py` provides multi-backend conversion
- Auto-detect array type (`detect_backend()`)
- Convert to NumPy (`as_numpy()`) for C++ backend
- Convert results back to original backend (`to_backend()`)
- Zero-copy optimization (when possible)

#### Nanobind / C++ Backend (Route A)

- The Python API uses a single backend: the compiled `polyfempy` C++ extension (nanobind). There are no separate backend modules (`backend_base` / `backend_dummy` / `backend_nanobind`); `solve.py` calls the C++ solver directly.
- `tensor.py` forces every array to become **CPU, C-contiguous NumPy** before it crosses the Python↔C++ boundary, matching nanobind’s zero-copy assumptions.
- `_ensure_i32()` and `Result` normalize cell connectivity to `int32`, aligning with nanobind/Eigen expectations on the C++ side.
- `solve.py` imports the `polyfempy` package and uses `cpp_backend_available()` / `cpp_backend_error()` for clear errors when the extension is not built.

**Details**: See [7. `tensor.py` - Tensor Conversion](#7-tensorpy---tensor-conversion)

### int32 Type Requirement

**Reasons**:
- C++ backend requires cell connectivity arrays to be `int32` type
- Memory efficiency: `int32` (4 bytes) saves memory compared to `int64` (8 bytes)
- Sufficient range: `int32` can represent ~2.1 billion vertices, enough for most scenarios

**Implementation**:
- `_ensure_i32()` function ensures correct type
- Automatic conversion in `solve()` and `Result`

### Version Compatibility Mechanism

**Reasons**:
- Support different versions of polyfempy C++ bindings
- Smooth upgrade path
- Backward compatibility

**Implementation**:
- `_first_attr()` finds available method names
- Support multiple method name variants (e.g., `set_mesh`, `set_mesh_data`, `load_mesh_from_points`)
- Graceful degradation

---

## 2. `solve.py` - Main Solve Function

### Design Philosophy

**Unified Interface + Version Compatibility**:
- Provide unified API interface, hide underlying implementation details
- Support different versions of polyfempy C++ bindings (via reflection)
- Auto-handle multi-backend arrays (see [Core Concepts - Multi-Backend Array Support](#multi-backend-array-support-numptorchjax))
- Version compatibility: find available method names via `_first_attr()`

### Core Function

#### `solve(vertices, cells, cfg, sidesets_func=None, dtype=None)`

**Functionality**:
- Unified solve entry point
- Auto-normalize user arrays (NumPy/Torch/JAX → NumPy)
- Build backend Settings/Problem
- Adapt to different polyfempy versions
- Apply boundary conditions
- Run solver
- Return Result object

**Parameters**:
- `vertices`: Vertex coordinates, shape (N, dim), supports NumPy/Torch/JAX (see [Core Concepts - Multi-Backend Array Support](#multi-backend-array-support-numptorchjax)). If None and cfg contains geometry, mesh will be loaded from files
- `cells`: Cell connectivity, shape (M, k), supports NumPy/Torch/JAX, auto-converted to int32 (see [Core Concepts - int32 Type Requirement](#int32-type-requirement)). If None and cfg contains geometry, mesh will be loaded from files
- `cfg`: SimulationConfig instance, dict, or str (file path). Supports full PolyFEM JSON configuration
- `sidesets_func`: Optional side set construction function
- `dtype`: Optional NumPy data type

**Returns**:
- `Result`: Result object containing solution and metadata

**Workflow**:

```
1. Process configuration (SimulationConfig/dict/str → SimulationConfig)
   - If file path, use from_json_file() to load
   - If dict, use from_json_dict() to convert
2. Check JSON mode (whether full JSON config and geometry exist)
   - If geometry exists and vertices/cells are None, use JSON mode
3. Normalize input arrays (if provided, NumPy/Torch/JAX → NumPy)
4. Import polyfempy (C++ bindings)
5. Construct solver (supports different versions)
6. Apply settings:
   - JSON mode: Use full JSON config directly, load mesh from files
   - Normal mode: Build Settings, set mesh arrays
7. Build basis and assemble (required for JSON mode)
8. Set side sets (optional)
9. Apply boundary conditions (normal mode, BC already in config for JSON mode)
10. Run solve
11. Get solution and return Result
```

### Helper Functions

#### `_first_attr(obj, *names)`

**Function**: Find first existing attribute name on object

**Purpose**: Version compatibility, support different API versions

**Example**:
```python
# Find settings or set_settings
name = _first_attr(solver, "settings", "set_settings")
if name:
    getattr(solver, name)(settings)
```

#### `_ensure_i32(cells)`

**Function**: Ensure cell array is int32 type

**Purpose**: Type normalization, ensure C++ backend compatibility (see [Core Concepts - int32 Type Requirement](#int32-type-requirement))

### Version Compatibility

Support different API versions via `_first_attr()` (see [Core Concepts - Version Compatibility Mechanism](#version-compatibility-mechanism)):

```python
# Support different method names
for name in ("set_mesh", "set_mesh_data", "load_mesh_from_points"):
    if hasattr(solver, name):
        fn = getattr(solver, name)
        try:
            fn(V_np, C_np)
            break
        except TypeError:
            try:
                fn(points=V_np, cells=C_np)
                break
            except Exception:
                pass
```

### Examples

- Example scripts: `examples/` at the repository root.

---

## 3. `config.py` - Configuration Class

> **Detailed Configuration Guide**: See [Configuration Guide](config-guide-en.md) - Complete guide on parameter input, validation, data flow, and extension.

### Design Philosophy

**Human-Friendly Config → Canonical Form → Backend Settings**:
- Provide intuitive configuration fields
- Auto-normalize aliases (PDE names, material parameters)
- Convert to backend Settings/Problem objects
- Support JSON serialization

### Core Class

#### `SimulationConfig`

**Functionality**:
- Store simulation configuration
- Normalize aliases (PDE, material parameters)
- Convert to backend Settings
- Support JSON serialization
- Validate configuration validity

**Attributes**:
- `pde`: PDE name (auto-normalized)
- `discr_order`: Discretization order
- `materials`: Material parameters (auto-normalized keys)
- `boundary_conditions`: Boundary conditions
- `extras`: Advanced options
- `selection`: Optional geometry selection object (`Selection` type) for selecting boundary conditions by geometric shapes
- `problem_type`: Optional predefined problem type (e.g., 'Gravity', 'Franke', 'TorsionElastic')
- `problem_params`: Parameter dict for predefined problems (e.g., `{'force': 0.1}` for Gravity problem)

**Main Methods**:

##### `canonicalized() -> SimulationConfig`

**Function**: Return normalized configuration copy

**Purpose**: Normalize aliases, ensure consistency

**Example**:
```python
cfg = SimulationConfig(pde="linear_elasticity")
cfg_canon = cfg.canonicalized()
# cfg_canon.pde == "LinearElasticity"
```

##### `to_settings() -> pf.Settings`

**Function**: Convert to backend Settings object

**Purpose**: Build backend configuration object

**Strategy**:
- PDE: 'Poisson' → `pf.GenericScalar()`
- PDE: 'LinearElasticity' → `pf.GenericTensor()`
- Materials: Set E/nu
- Advanced options: Pass via `set_advanced_option`
- **Predefined problems**: If `problem_type` is specified, prioritize predefined problems (loaded from `pf` or `polyfempy.legacy.Problems`)
- **Selection handling**: If config contains `selection`, convert to dict and store in `settings._selection` for later use
- **Version compatibility**: Auto-adapt to different polyfempy API versions (supports `set_problem`, `set_pde`, etc.)
- **Graceful degradation**: If C++ backend unavailable, use `_DummySettings` placeholder to keep API usable

##### `to_json_str() -> str`

**Function**: Serialize to JSON string

**Purpose**: Configuration persistence

##### `to_dict() -> dict`

**Function**: Convert SimulationConfig to dictionary representation

**Purpose**: Get dictionary form of configuration for backend use or serialization

**Behavior**:
- If config contains full JSON (`extras["_full_json_config"]`), directly return that full config
- Otherwise, construct dict from fields, including normalized fields and optional fields (extras, problem_type, problem_params, selection)
- **Parameter promotion**: Extract common parameters (e.g., `max_iters`, `random_seed`) from `extras` and promote to top level, with validation and type conversion

**Example**:
```python
cfg = SimulationConfig.linear_elasticity(2100, 0.3)
config_dict = cfg.to_dict()
# Returns dict containing pde, discr_order, materials, boundary_conditions, etc.

# Parameter promotion example
cfg = SimulationConfig(extras={"max_iters": "10", "random_seed": "42"})
d = cfg.to_dict()
# d["max_iters"] = 10 (int, promoted from extras to top level)
# d["random_seed"] = 42 (int, promoted from extras to top level)
```

**Details**: See [Configuration Guide](config-guide-en.md) sections on "Data Flow" and "Parameter Validation"

##### `from_json_str(s: str) -> SimulationConfig`

**Function**: Deserialize from JSON string

**Purpose**: Configuration loading

##### `from_json_dict(d: dict) -> SimulationConfig`

**Function**: Create configuration from full PolyFEM JSON dictionary

**Purpose**: Load full PolyFEM JSON configuration (supports all parameters)

**Supported Parameters**:
- `geometry`: Mesh files, transformations, selections
- `materials`: All material types and parameters
- `boundary_conditions`: All boundary condition types
- `time`: Transient settings (t0, tend, dt, integrator)
- `contact`: Contact settings (enabled, dhat, mu, epsv, etc.)
- `solver`: Linear/nonlinear solver settings
- `output`: Paraview, JSON output settings
- `space`: Discretization order (supports list format, e.g., `[{"id": 2, "order": 2}]`)
- `common`: JSON references (auto-merged)

**Implementation Strategy**:
- Full JSON saved in `extras["_full_json_config"]`
- Known fields (pde, discr_order, materials) also extracted for use
- Support deep nested merging of `common.json`
- **PDE auto-inference**: If PDE not specified and material type is NeoHookean or SaintVenant, auto-infer as `NonLinearElasticity`
- **Material format support**: Support dict and array formats for material definitions (take first material if array format)
- **discr_order format support**: Support scalar, list, and nested dict formats (e.g., `space.discr_order`)

**Example**:
```python
import json
with open("config.json") as f:
    config_dict = json.load(f)
cfg = SimulationConfig.from_json_dict(config_dict)
# Full config accessible via cfg.to_dict() or cfg.extras["_full_json_config"]
```

##### `from_json_file(filepath: str) -> SimulationConfig`

**Function**: Load configuration from JSON file

**Purpose**: Directly load PolyFEM JSON configuration from file

**Implementation**:
- Auto-handle `common.json` references
- Auto-merge referenced common.json files
- Support deep nested merging

**Example**:
```python
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")
```

##### `validate() -> None`

**Function**: Validate configuration validity

**Checks**:
- `discr_order` must be positive integer
- `materials['E']` and `materials['nu']` must be numbers

##### Convenience Factory Methods

**Basic PDE Methods**:
- `linear_elasticity(E, nu, order=1)`: Linear elasticity problem
- `poisson(order=1)`: Poisson problem

**Predefined Problem Methods**:
- `gravity(force=0.1, E=None, nu=None, order=1)`: Gravity problem
- `franke(order=1)`: Franke problem (scalar, has exact solution)
- `torsion(axis_coordinate=2, n_turns=0.5, ...)`: Torsion problem (3D)
- `flow(inflow=1, outflow=3, ...)`: Flow problem (inflow/outflow)
- `driven_cavity(order=1)`: Driven cavity problem
- `flow_with_obstacle(U=1.5, time_dependent=True, order=1)`: Flow with obstacle

These are class methods (`@classmethod`), can be called directly via `SimulationConfig.method_name()` without creating an instance first.

---

## 4. `selection.py` - Geometry Selection Tool

### Design Philosophy

**Geometry Selection Instead of ID Selection**: Allow users to select boundary conditions via geometric shapes (sphere, box, plane) without needing to know specific sideset IDs in mesh files. Particularly useful when mesh files don't have correct sideset markers.

### Core Class

#### `Selection`

**Functionality**:
- Select boundary conditions via geometric shapes
- Support body and sideset selection
- Support multiple geometric shapes (sphere, box, plane)

**Methods**:
- Body selection: `select_body_with_sphere()`, `select_body_with_box()`, `select_body_with_axis_plane()`, `select_body_with_plane()`
- Sideset selection: `select_sideset_with_sphere()`, `select_sideset_with_box()`, `select_sideset_with_axis_plane()`, `select_sideset_with_plane()`
- Conversion: `to_dict()`, `to_json_str()`

**Example**:
```python
from polyfempy.api import SimulationConfig, Selection

selection = Selection()
selection.select_sideset_with_sphere(id=1, center=[0, 0, 0], radius=1.0)

cfg = SimulationConfig(
    selection=selection,
    boundary_conditions={"dirichlet_boundary": [{"id": 1, "value": [0, 0]}]}
)
```

---

## 5. `result.py` - Result Container

### Design Philosophy

**Unified Container + Multi-Backend Support**:
- Internal storage as NumPy arrays (C-contiguous)
- Support conversion back to original backend (see [Core Concepts - Multi-Backend Array Support](#multi-backend-array-support-numptorchjax))
- Provide field management
- Support VTK export

### Core Class

#### `Result`

**Functionality**:
- Store mesh and point field data
- Support multi-backend array conversion
- Provide field management
- Support VTK export

**Attributes**:
- `backend`: Original backend name ('numpy'|'torch'|'jax')
- `vertices`: Vertex coordinates, shape (N, dim)
- `cells`: Cell connectivity, shape (M, k), auto-converted to int32
- `fields`: Field dictionary, e.g., {'u': (N, dim)}
- `meta`: Metadata dictionary

**Main Methods**:
- `field(name)`: Get field value
- `set_field(name, value)`: Set field value
- `remove_field(name)`: Remove field
- `as_numpy()`: Normalize to NumPy (idempotent)
- `to_backend(include_mesh=False)`: Convert back to original backend
- `magnitude(name, out_name=None, eps=0.0)`: Compute vector field magnitude
- `to_vtk(path)`: Export to VTK format
- `summary()`: Return result summary
- `field_names()`: Return all field names

---

## 6. `batch.py` - Batch Processing

### Design Philosophy

**Sequential Processing + Error Isolation + Order Preservation**:
- Process all tasks sequentially
- Error isolation: One task failure doesn't affect others
- Order preservation: Result order matches input order

### Core Function

#### `batch_solve(jobs) -> list[Result]`

**Parameters**:
- `jobs`: List of tasks, each task is:
  - 3-tuple: `(V, C, cfg)`
  - 4-tuple: `(V, C, cfg, kwargs_dict)`

**Returns**:
- `list[Result]`: Result list, order matches input

**Error Handling**:
- Tasks are run sequentially; if a task raises, that exception propagates (no per-job error isolation in the current implementation).

---

## 7. `tensor.py` - Tensor Conversion

### Design Philosophy

**Multi-Backend Support + Zero-Copy Optimization** (see [Core Concepts - Multi-Backend Array Support](#multi-backend-array-support-numptorchjax)):
- Support NumPy/Torch/JAX arrays
- Auto-detect backend
- Zero-copy conversion (when possible)
- Enforce C-contiguous layout

### Why Do We Need tensor.py? Even Though nanobind Supports Zero Copy

**Core Reason**: nanobind's zero copy **only works for NumPy arrays**, but users may use Torch or JAX arrays.

**nanobind Limitations**:
- ✅ Supports: NumPy arrays → C++ (zero copy)
- ❌ Doesn't support: Torch tensors → C++
- ❌ Doesn't support: JAX arrays → C++
- ❌ Doesn't support: GPU tensors

**tensor.py Role**:
1. **Multi-backend support**: Convert Torch/JAX arrays to NumPy arrays
2. **Device handling**: Move GPU tensors to CPU
3. **Memory layout**: Ensure C-contiguous layout
4. **Return conversion**: Convert results back to user's original backend

**Data Flow**:
```
User Torch Tensor
    ↓ tensor.py: Convert to NumPy (zero copy, CPU + contiguous)
NumPy Array (C-contiguous)
    ↓ nanobind: zero copy NumPy → C++
C++ Eigen Matrix
    ↓ computation
C++ Result
    ↓ nanobind: zero copy C++ → NumPy
NumPy Array
    ↓ tensor.py: Convert back to Torch (zero copy)
User Torch Tensor
```

### Core Functions

- `detect_backend(x) -> str`: Detect array backend ('numpy' | 'torch' | 'jax')
- `as_numpy(x, dtype=None) -> (np.ndarray, str)`: Convert to NumPy array
- `to_backend(arr, backend) -> array`: Convert back to specified backend
- `from_numpy(arr, backend) -> array`: Convert from NumPy to specified backend

### Zero-Copy Optimization

**Important Note**: Zero-copy happens at two different levels:

1. **PyTorch Zero-Copy** (Torch ↔ NumPy):
   - This is PyTorch's own feature, not nanobind's
   - `tensor.numpy()`: When tensor is on CPU and contiguous, returned NumPy array **shares memory** with tensor
   - `torch.from_numpy()`: Create tensor from NumPy array, **shares memory** (requires C-contiguous)

2. **nanobind Zero-Copy** (NumPy ↔ C++):
   - nanobind supports zero-copy between NumPy arrays and C++
   - But doesn't support Torch tensors directly to C++

**Complete Data Flow**:
```
User Torch Tensor
    ↓ PyTorch zero-copy: t.numpy() (CPU + contiguous)
NumPy Array (shared memory)
    ↓ nanobind zero-copy: NumPy → C++
C++ Eigen Matrix
    ↓ computation
C++ Result
    ↓ nanobind zero-copy: C++ → NumPy
NumPy Array
    ↓ PyTorch zero-copy: torch.from_numpy()
User Torch Tensor
```

---

## Example Architecture

Examples are located in the **repository root** under `examples/`, not inside `polyfempy/api/`.

### Example Files

- **`examples/python_config_5_cubes.py`** – Full-featured example: multi-body geometry (5 cubes + plane), contact, NeoHookean material, time stepping, solver/output configuration, result read/write.
- **`examples/contact_5_cubes.py`** – Contact example variant.
- **`examples/README_设计说明.md`** – Design notes (Chinese).

### Running Examples

From the repository root:

```bash
python examples/python_config_5_cubes.py
python examples/contact_5_cubes.py
```

---

## Design Patterns

1. **Facade Pattern**: `solve()` – Single entry point; hides config normalization, C++ invocation, and result construction.
2. **Factory Pattern**: `SimulationConfig` and config classes (e.g. `NeoHookean`, `Geometry`, `Solver`) – Build JSON/config for the C++ backend.
3. **Direct C++ binding**: `solve.py` calls the compiled `polyfempy` extension (Route A); there is no separate backend SPI or dummy backend.

---

## Data Flow

### Solve Flow

```
User Code
    ↓
solve(vertices, cells, cfg)
    ↓
1. Normalize inputs (tensor.py)
    - as_numpy(vertices) → V_np, backend
    - as_numpy(cells) → C_np, _
    ↓
2. Build config / JSON (config.py)
    - cfg.to_dict() or full JSON from cfg
    ↓
3. Call C++ (solve.py)
    - Import polyfempy; check cpp_backend_available()
    - Build solver, set mesh or load from geometry, run solve
    - Parse (sol, pressure) into result dict
    ↓
4. Build result (result.py)
    - Result(backend, vertices, cells, fields, meta)
    ↓
5. Return result
    - result.to_backend() → Convert back to original backend
    ↓
User Code
```

### Configuration Flow

**Method 1: Programmatic Configuration**
```
User Config
    ↓
SimulationConfig(pde="linear_elasticity", ...)
    ↓
cfg.canonicalized()
    - Normalize PDE name
    - Normalize material parameters
    ↓
cfg.to_settings()
    - Create pf.Settings
    - Set PDE type
    - Set material parameters
    - Set advanced options
    ↓
pf.Settings Object
```

**Method 2: JSON File Configuration**
```
JSON File (may contain common.json reference)
    ↓
SimulationConfig.from_json_file(filepath)
    - Load JSON file
    - Detect and merge common.json (deep recursive merge)
    - Extract known fields (pde, discr_order, materials)
    - Full JSON saved in extras["_full_json_config"]
    ↓
SimulationConfig Instance (contains full JSON)
    ↓
solve() detects JSON mode
    - If geometry exists and no vertices/cells, use JSON mode
    - Use full JSON config directly
    - Load mesh from files
    ↓
Solver Configuration Complete
```

---

## Summary

### Core Design Principles

1. **Unified Interface**: Provide simple API, hide complexity
2. **Multi-Backend Support**: Support NumPy/Torch/JAX arrays
3. **Version Compatibility**: Support different versions of C++ bindings
4. **Order preservation**: `batch_solve` returns results in the same order as input jobs
5. **Deterministic Output**: Same input produces same output
### File Responsibilities

- **`solve.py`**: Main solve function; calls C++ extension directly (Route A)
- **`config.py`**: Configuration and config classes (materials, BCs, geometry, solver, output, etc.)
- **`result.py`**: Result container, multi-backend support, meshio/VTK read/write
- **`selection.py`**: Geometry-based selection for boundary conditions
- **`batch.py`**: Batch processing (sequential, order-preserving)
- **`tensor.py`**: Tensor conversion (NumPy/Torch/JAX)
- **`io.py`**: Mesh I/O (`read_mesh`, `Mesh`; requires meshio)

