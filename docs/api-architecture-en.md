# PolyFEM Python API Architecture

This document explains all files in the `polyfempy/api/` directory, including design philosophy, function purposes, and examples.

> Note (Route A): Python should always `import polyfempy as pf` (stable C++ extension module name).
> nanobind vs pybind11 is a build-time choice and must not affect imports. The C++ binding’s
> `Solver.solve()` returns `(sol, pressure)`, so Python callers must capture and parse it.

## Directory Structure

```
polyfempy/api/
├── __init__.py              # Module entry, exports main API
├── solve.py                 # Main solve function (unified entry point)
├── config.py                # Configuration class (SimulationConfig)
├── result.py                # Result container (Result)
├── errors.py                # Error handling (unified error model)
├── backend_base.py          # Backend SPI definition (interface contract)
├── backend_dummy.py         # Dummy backend implementation (for testing)
├── backend_nanobind.py      # Nanobind backend adapter (C++ connection)
├── batch.py                 # Batch processing (batch_solve)
├── tensor.py                # Tensor conversion (multi-backend support)
└── examples/                # API example code
    ├── run_dummy_elasticity.py
    ├── run_elasticity.py
    ├── load_from_json.py
    ├── parameter_sweep.py
    ├── batch_processing.py
    ├── with_callbacks.py
    └── README.md
```

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
- **`batch_solve`**: Batch solve function with error isolation

### Examples

- Example files: `polyfempy/api/examples/`
- Example content: See [Examples README](../polyfempy/api/examples/README.md)

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

#### Nanobind Compatibility Work

- `tensor.py` forces every array to become **CPU, C-contiguous NumPy** before it crosses the Python↔C++ boundary, matching nanobind’s zero-copy assumptions
- `_ensure_i32()` and `Result` normalize cell connectivity to `int32`, aligning with nanobind/Eigen expectations on the C++ side
- `backend_nanobind.py` (deprecated) no longer imports `polyfem_nb/solve_cpp`; Route A uses `polyfempy` directly
- `solve.py` directly detects the nanobind-built `polyfempy` module and relies on its zero-copy plumbing plus `Solver/State` APIs
- All scripts in `polyfempy/api/examples/` operate on NumPy data, making it easy to validate the nanobind data path end to end

**Details**: See [11. `tensor.py` - Tensor Conversion](#11-tensorpy---tensor-conversion)

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

- Example files: `polyfempy/api/examples/`
- Example content: See [Examples README](../polyfempy/api/examples/README.md)

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

## 6. `errors.py` - Error Handling

### Design Philosophy

**Unified Error Model + Clear Prefixes**:
- All errors have clear prefixes (INPUT:/CALLBACK:/BACKEND:)
- Error types correspond to error causes
- Provide clear error messages

### Core Functions

- `raise_input_error(msg)`: Raise input error (ValueError, prefix: INPUT:)
- `raise_callback_type_error(msg)`: Raise callback error (TypeError, prefix: CALLBACK:)
- `raise_backend_error(msg)`: Raise backend error (RuntimeError, prefix: BACKEND:)

---

## 7. `backend_base.py` - Backend SPI Definition

### Design Philosophy

**Interface Contract + Documentation**:
- Define backend interface contract (SPI)
- Provide detailed documentation
- Ensure all backend implementations are consistent

### Core Function

#### `solve_impl(V, C, settings, callbacks) -> dict`

**Function**: Backend SPI interface (documentation function)

**Input Contract**:
- `V`: np.ndarray, shape (N, dim), dtype float64, C-contiguous
- `C`: np.ndarray, shape (M, k), dtype int32, C-contiguous
- `settings`: dict from SimulationConfig.to_dict()
- `callbacks`: dict[str, callable] or None

**Output Contract**:
- Must return dict with keys:
  - `u`: np.ndarray, shape (N, dim), dtype float64, C-contiguous
  - `strain`: np.ndarray or None
  - `stress`: np.ndarray or None
  - `meta`: dict with required keys (backend, iters, residual, seed)

**Backend Responsibilities**:
1. Input validation (though caller already validates)
2. Callback handling (call in order: before_solve → after_iter×K → after_solve)
3. Deterministic output (if random_seed provided)
4. Return result dict conforming to contract

---

## 8. `backend_dummy.py` - Dummy Backend Implementation

### Design Philosophy

**Strict Validation + Deterministic Output + Callback Testing**:
- Strictly validate inputs (dtype, shape, contiguity)
- Generate deterministic pseudo-random output
- Correctly handle callback lifecycle
- Unified error model

### Core Function

#### `solve_impl(V, C, settings, callbacks) -> dict`

**Implementation Strategy**:
1. Input validation: Strictly validate V and C dtype, shape, contiguity
2. Parse settings: Get max_iters, random_seed from settings
3. Callback handling: Call callbacks in order (before_solve → after_iter×K → after_solve)
4. Generate output: Use deterministic RNG to generate pseudo-random displacement field
5. Return result: Return dict conforming to SPI contract

**Deterministic Output**:
```python
rng = np.random.RandomState(random_seed)
u = rng.normal(loc=0.0, scale=1e-3, size=(N, dim)).astype(np.float64)
```

**Residual Model**:
```python
residual = 1e-3 / (i + 1)  # Linear decrease
```

---

## 9. `backend_nanobind.py` - Nanobind Backend Adapter

### Design Philosophy

**Adapter Pattern + Graceful Degradation**:
- Try to import C++ backend
- If unavailable, provide clear error message
- Forward calls to C++ backend

### Core Function

#### `solve_impl(V, C, settings, callbacks) -> dict`

**Implementation Strategy**:
1. Check availability: Try to import `polyfempy as pf`
2. Error handling: If unavailable, raise a clear error (build/install the C++ extension first)
3. Call C++: construct `pf.Solver()/pf.State()`, call `solver.solve()`, and parse `(sol, pressure)`

---

## 10. `batch.py` - Batch Processing

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
- If task fails, result position contains Exception object
- Other tasks continue execution
- Order preserved: Result order matches input order

---

## 11. `tensor.py` - Tensor Conversion

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

### Example File Structure

```
polyfempy/api/examples/
├── run_dummy_elasticity.py      # Basic Dummy backend example
├── run_elasticity.py            # Minimal 2D linear elasticity example
├── load_from_json.py            # Load configuration from JSON file
├── parameter_sweep.py           # Parameter sweep (sensitivity study)
├── batch_processing.py          # Batch processing (error isolation)
├── with_callbacks.py            # Monitor solve with callbacks
└── README.md                    # Example documentation
```

### Example Categories

1. **Basic Examples**: Show basic API usage
   - `run_dummy_elasticity.py` - Simplest usage example
   - `run_elasticity.py` - Linear elasticity problem example

2. **Configuration Examples**: Show different configuration methods
   - `load_from_json.py` - Load full configuration from JSON file

3. **Advanced Usage Examples**: Show advanced features and best practices
   - `parameter_sweep.py` - Parameter sweep and sensitivity analysis
   - `batch_processing.py` - Batch processing and error isolation
   - `with_callbacks.py` - Monitor progress with callbacks

### Running Examples

All examples can be run via:

```bash
python -m polyfempy.api.examples.run_dummy_elasticity
python -m polyfempy.api.examples.parameter_sweep
python -m polyfempy.api.examples.batch_processing
python -m polyfempy.api.examples.with_callbacks
python -m polyfempy.api.examples.load_from_json
```

---

## Design Patterns

1. **Adapter Pattern**: `backend_nanobind.py` - Adapt C++ backend to Python API
2. **Strategy Pattern**: Backend switching (dummy vs nanobind)
3. **Factory Pattern**: `SimulationConfig.to_settings()` - Create backend Settings objects based on config
4. **Facade Pattern**: `solve()` function - Hide complexity, provide simple interface
5. **Singleton Pattern**: Error handling functions - Provide unified error handling interface

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
2. Build settings (config.py)
    - cfg.to_settings() → settings
    ↓
3. Select backend (backend_base.py)
    - backend == "dummy" → backend_dummy.solve_impl()
    - backend == "nanobind" → backend_nanobind.solve_impl()
    ↓
4. Backend implementation
    - Input validation
    - Callback handling
    - Run solve
    - Return result dict
    ↓
5. Build result (result.py)
    - Result(backend, vertices, cells, fields, meta)
    ↓
6. Return result
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
4. **Error Isolation**: Errors in batch processing don't affect each other
5. **Deterministic Output**: Same input produces same output
6. **Clear Errors**: All errors have clear prefixes and messages

### File Responsibilities

- **`solve.py`**: Main solve function, unified entry point
- **`config.py`**: Configuration class, normalize configuration
- **`result.py`**: Result container, multi-backend support
- **`errors.py`**: Error handling, unified error model
- **`backend_base.py`**: Backend SPI definition, interface contract
- **`backend_dummy.py`**: Dummy backend implementation, for testing
- **`backend_nanobind.py`**: Nanobind backend adapter, C++ connection
- **`batch.py`**: Batch processing, error isolation
- **`tensor.py`**: Tensor conversion, multi-backend support

