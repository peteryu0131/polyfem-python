# SimulationConfig Configuration Guide

This document provides a comprehensive guide to using `SimulationConfig`, including parameter input, validation, data flow, and extension.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Flow Process](#data-flow-process)
3. [Parameter Input Methods](#parameter-input-methods)
4. [Class-Based Configuration Details](#class-based-configuration-details)
5. [Parameter Validation Mechanism](#parameter-validation-mechanism)
6. [JSON Parameter Support](#json-parameter-support)
7. [Practical Application Examples](#practical-application-examples)
8. [Adding New Parameters](#adding-new-parameters)

---

## Quick Start

### Basic Usage

```python
from polyfempy.api import SimulationConfig, solve
import numpy as np

# Create a simple configuration
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}
)

# Use the configuration
V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
result = solve(V, C, cfg)
```

### Loading from JSON File

```python
# Load complete configuration from JSON file
cfg = SimulationConfig.from_json_file("config.json")
result = solve(vertices=None, cells=None, cfg=cfg)
```

---

## Data Flow Process

### Why Are There So Many Dictionaries?

In the API, there are several different dictionaries used at different stages:

1. **`SimulationConfig.extras`** - User-input extra parameters
2. **Dictionary returned by `SimulationConfig.to_dict()`** - Configuration passed to the solver (C++ extension)
3. **`settings` dictionary (passed by `solve()` to C++ extension)** - Configuration actually used by the solver

### Data Flow Process

#### Stage 1: User Creates Configuration

```python
# User creates SimulationConfig
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}  # ← Placed in extras
)
```

**Why place in `extras`?**
- `SimulationConfig` is a dataclass with fixed fields: `pde`, `discr_order`, `materials`, `boundary_conditions`, `extras`, `selection`, `problem_type`, `problem_params`
- `max_iters` and `random_seed` are not direct fields of `SimulationConfig`
- So they need to be placed in the `extras` "miscellaneous" dictionary

#### Stage 2: Convert to Dictionary (`to_dict()`)

```python
# Call to_dict() to convert to dictionary
settings_dict = cfg.to_dict()
# Result:
# {
#     "pde": "linear_elasticity",
#     "discr_order": 1,
#     "materials": {"E": 1e6, "nu": 0.3},
#     "boundary_conditions": {},
#     "extras": {"max_iters": 10, "random_seed": 42},
#     "max_iters": 10,        # ← Promoted from extras to top level
#     "random_seed": 42       # ← Promoted from extras to top level
# }
```

**Why promote to top level?**
- When `solve()` passes the config (from `cfg.to_dict()`) to the C++ extension or uses it to build JSON, the C++ side expects common parameters like `max_iters` and `random_seed` at the top level.
- If they were only in `extras`, the C++ or Python glue would need `settings["extras"]["max_iters"]`, which is inconvenient and error-prone.

#### Stage 3: Pass to Solver

```python
# Inside solve(): get config dict and call C++ extension
settings = cfg.to_dict()  # Get dictionary (with promoted top-level params)
# solve() uses settings to build JSON or Settings, then calls the polyfempy C++ extension
result = solve(V, C, cfg)  # User only calls solve(); no direct exposure to settings
```

**How the solver uses it:**
- `solve()` in `solve.py` calls the compiled `polyfempy` C++ extension directly (Route A); there are no separate backend modules.
- If the C++ or internal logic needs `max_iters` or `random_seed`, it can read from the top level of the config dict: `settings.get("max_iters", 10)`.

### Complete Data Flow Example

```python
# 1. User creates configuration
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}
)

# 2. Convert to dictionary (called automatically internally)
settings = cfg.to_dict()
# settings = {
#     "pde": "linear_elasticity",
#     "discr_order": 1,
#     "materials": {"E": 1e6, "nu": 0.3},
#     "boundary_conditions": {},
#     "extras": {"max_iters": 10, "random_seed": 42},  # Original extras preserved
#     "max_iters": 10,        # Promoted to top level for solver convenience
#     "random_seed": 42       # Promoted to top level for solver convenience
# }

# 3. solve() calls C++ extension
result = solve(V, C, cfg)  # Internally calls cfg.to_dict() to get settings, then passes to C++

# 4. Solver uses settings
# solve.py passes settings to the C++ extension; max_iters/random_seed read from top level if needed
```

### Design Rationale

**Design Rationale 1: Type Safety vs Flexibility**
- **`SimulationConfig` is type-safe**: Has fixed fields, IDE can autocomplete
- **`extras` provides flexibility**: Can store arbitrary extra parameters without modifying class definition
- **`to_dict()` provides conversion**: Converts type-safe object to flexible dictionary

**Design Rationale 2: Compatibility with C++ Extension**
- **Solver expects a simple dictionary**: `solve()` passes config as a `dict` to the C++ extension or uses it to build JSON; no need to pass complex Python objects
- **Promote common parameters**: `max_iters` and `random_seed` are common parameters, promoting to top level is convenient
- **Preserve original data**: `extras` is still preserved in case other extra parameters need to be accessed

**Design Rationale 3: Support Full JSON Configuration**
```python
# If loading from full JSON
cfg = SimulationConfig.from_json_file("config.json")
# Full JSON stored in extras["_full_json_config"]
# to_dict() will directly return full JSON, not constructed dictionary
```

---

## Parameter Input Methods

### Method 1: Using Classes (Recommended - IDE Autocomplete Support)

To provide better IDE autocomplete support and type checking, `SimulationConfig`'s main parameters support class form.

**Basic Example:**
```python
from polyfempy.api.config import SimulationConfig, Material, BoundaryConditions, GravityParams

# Use Material class
material = Material(E=2100, nu=0.3, rho=1.0)
cfg = SimulationConfig(materials=material)

# Use BoundaryConditions class
bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])
cfg = SimulationConfig(boundary_conditions=bc)

# Use ProblemParams classes
gravity_params = GravityParams(force=0.1)
cfg = SimulationConfig(problem_type="Gravity", problem_params=gravity_params)
```

**Advantages**:
- ✅ IDE autocomplete: When typing `material.`, IDE will suggest all available attributes
- ✅ Type checking: IDE can validate parameter types
- ✅ Error prevention: Spelling errors will be caught by IDE
- ✅ Clear code: `material.E` is clearer than `materials["E"]`

**Detailed Documentation**: Please refer to the Config Classes Guide for all available Classes, convenience methods, and complete examples.

### Method 2: Using Dict (Backward Compatible)

```python
# Old way still supported (but IDE cannot autocomplete)
cfg = SimulationConfig(
    pde="linear_elasticity",
    discr_order=2,
    materials={"E": 1e6, "nu": 0.3},  # ⚠️ IDE cannot suggest key names
    boundary_conditions={
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],  # ⚠️ IDE cannot suggest
        "rhs": [1.0, 0.0]
    },
    problem_params={"force": 0.1}  # ⚠️ IDE cannot suggest key names
)
```

**Note**: Although dict method is still supported, it's recommended to use class method for better development experience.

### Method 3: Input Extra Parameters via extras

```python
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={
        # Common parameters (will be validated and promoted to top level)
        "max_iters": 10,
        "random_seed": 42,
        
        # Other JSON parameters (kept in extras)
        "solver": {
            "linear": {"max_iter": 1000, "tolerance": 1e-6},
            "nonlinear": {"max_iter": 50}
        },
        "time": {
            "t0": 0.0,
            "tend": 1.0,
            "dt": 0.01
        }
    }
)
```

### Method 4: Load from JSON File (Recommended for Complete Configuration)

```python
# Load complete configuration from JSON file
cfg = SimulationConfig.from_json_file("config.json")
# All parameters are saved in extras["_full_json_config"]
```

### Method 5: Load from Dictionary

```python
config_dict = {
    "pde": "LinearElasticity",
    "materials": [{"type": "LinearElasticity", "E": 1e6, "nu": 0.3}],
    "solver": {...},
    "time": {...}
}
cfg = SimulationConfig.from_json_dict(config_dict)
```

---

## Class-Based Configuration Details

To provide better IDE autocomplete support and type checking, `SimulationConfig`'s main parameters support class form (such as `Material`, `BoundaryConditions`, `GravityParams`, etc.).

**Detailed Documentation**: Please refer to the Config Classes Guide for:
- Why use Classes instead of Dict
- All available Classes (Material, BoundaryConditions, ProblemParams, etc.)
- Convenience methods and Convenience Factories
- Complete usage examples

---

## Parameter Validation Mechanism

### Validation Levels

#### 1. SimulationConfig Level Validation

**Parameters that will be checked:**
- `discr_order`: Must be a positive integer (via `validate()` method)
- `materials['E']` and `materials['nu']`: Must be numbers (via `validate()` method)

**Examples:**
```python
# ✅ Will be checked
cfg = SimulationConfig(discr_order=-1)  # ValueError: discr_order must be positive
cfg = SimulationConfig(materials={"E": "abc"})  # ValueError: materials['E'] must be a number
```

#### 2. to_dict() Level Validation

**Current Behavior:**
- `to_dict()` **validates and converts** common parameters in `extras`
- `max_iters`: Must be a positive integer, string `"10"` will be automatically converted to integer `10`
- `random_seed`: Must be an integer or None, string `"42"` will be automatically converted to integer `42`
- Invalid values will immediately raise `ValueError` with clear error messages

**Examples:**
```python
# ✅ String will be automatically converted to integer
cfg = SimulationConfig(extras={"max_iters": "10"})
d = cfg.to_dict()
print(d["max_iters"])  # 10 (integer)
print(type(d["max_iters"]))  # <class 'int'>

# ❌ Invalid string will immediately raise error
cfg = SimulationConfig(extras={"max_iters": "abc"})
d = cfg.to_dict()  # ValueError: extras['max_iters'] must be a positive integer, got 'abc'

# ❌ Negative number will immediately raise error
cfg = SimulationConfig(extras={"max_iters": -5})
d = cfg.to_dict()  # ValueError: extras['max_iters'] must be a positive integer, got -5
```

### Validation Rules

1. **`max_iters`**:
   - Must be a positive integer
   - String `"10"` will be automatically converted to integer `10`
   - Negative numbers or invalid strings will raise `ValueError`

2. **`random_seed`**:
   - Must be an integer or `None`
   - String `"42"` will be automatically converted to integer `42`
   - Invalid strings will raise `ValueError`

3. **Other JSON Parameters** (`solver`, `time`, `output`, `contact`, `geometry`, etc.):
   - Can be input via `extras`
   - Will not be validated (type checking)
   - Kept in `extras`, not promoted to top level
   - If using full JSON mode (loading from file), all parameters will be preserved
   - Requires the C++ extension (polyfempy build) to use

### Handling Unknown Parameters

**Design Behavior:**
- Unknown parameters will not be validated
- Kept in `extras`, but not promoted to top level
- Will not raise error (allows custom parameters)

**Example:**
```python
# User made a spelling error
cfg = SimulationConfig(extras={"max_iter": 10})  # Should be max_iters

# Will not raise error, but parameter is ignored (uses default value)
settings = cfg.to_dict()
# settings["max_iters"] doesn't exist, uses default value 10
```

**Why this design?**
- Allows users to add custom parameters to `extras`, which may be used by other tools
- Only validates known critical parameters (`max_iters`, `random_seed`)

---

## JSON Parameter Support

### Complete Parameter List for PolyFEM JSON Configuration

According to `from_json_dict()` documentation, PolyFEM JSON configuration supports all the following parameters:

#### 1. Basic Parameters
- `pde`: PDE type ("Poisson", "LinearElasticity", "NonLinearElasticity", etc.)
- `discr_order`: Discretization order (1, 2, ...)
- `space`: Space configuration (includes `discr_order`, etc.)

#### 2. Material Parameters
- `materials`: Material configuration (array or dict format)
  - `type`: Material type ("LinearElasticity", "NeoHookean", "SaintVenant", etc.)
  - `E`: Young's modulus
  - `nu`: Poisson's ratio
  - `rho`: Density
  - `id`: Material ID

#### 3. Boundary Conditions
- `boundary_conditions`: Boundary condition configuration
  - `dirichlet_boundary`: Dirichlet boundary conditions
  - `neumann_boundary`: Neumann boundary conditions
  - `pressure`: Pressure boundary conditions
  - `rhs`: Body forces and right-hand side

#### 4. Solver Parameters
- `solver`: Solver configuration
  - `linear`: Linear solver configuration
    - `max_iter`: Maximum iterations
    - `tolerance`: Tolerance
  - `nonlinear`: Nonlinear solver configuration
    - `max_iter`: Maximum iterations
    - `tolerance`: Tolerance
  - `max_threads`: Maximum threads

#### 5. Time Parameters (Transient Problems)
- `time`: Time configuration
  - `t0`: Initial time
  - `tend`: End time
  - `dt`: Time step
  - `time_steps`: Number of time steps
  - `integrator`: Integrator type ("ImplicitEuler", "ExplicitEuler", etc.)

#### 6. Output Parameters
- `output`: Output configuration
  - `directory`: Output directory
  - `paraview`: ParaView output configuration
  - `json`: JSON output configuration

#### 7. Contact Parameters
- `contact`: Contact configuration
  - `enabled`: Whether contact is enabled
  - `dhat`: Contact distance threshold
  - `mu`: Friction coefficient
  - `epsv`: Viscosity parameter

#### 8. Geometry Parameters
- `geometry`: Geometry configuration
  - `mesh`: Mesh file path
  - `transformations`: Transformations
  - `selections`: Selections

### How to Input These Parameters

These JSON parameters can be input in the following ways (for detailed explanation, see [Parameter Input Methods](#parameter-input-methods) section):

- **Load from JSON file**: `SimulationConfig.from_json_file("config.json")`
- **Input via extras**: `SimulationConfig(extras={"solver": {...}, "time": {...}})`
- **Load from dictionary**: `SimulationConfig.from_json_dict(config_dict)`

### to_dict() Behavior

1. **If there is full JSON configuration** (`extras["_full_json_config"]`):
   ```python
   # Directly return full JSON, containing all parameters
   return dict(self.extras["_full_json_config"])
   ```

2. **If there is no full JSON configuration**:
   ```python
   # Only construct basic fields
   result = {
       "pde": ...,
       "discr_order": ...,
       "materials": ...,
       "boundary_conditions": ...,
   }
   
   # Promote common parameters from extras to top level
   if "max_iters" in extras:
       result["max_iters"] = extras["max_iters"]  # Validate and convert
   
   # Other extras parameters kept in extras
   result["extras"] = dict(extras)
   ```

---

## Practical Application Examples

### Compatibility with polyfem-data Examples

✅ **API Design Success**: All 86 JSON configuration files from `polyfem-data/contact/examples/2D` and `3D` can be successfully loaded and configured using the new API.
- **2D Examples**: 28/28 successful
- **3D Examples**: 58/58 successful

### Test Results

#### Configuration Loading
- **86/86 successful** - All JSON files can be loaded into `SimulationConfig`
  - 2D: 28/28 successful
  - 3D: 58/58 successful
- All parameters are saved in `extras["_full_json_config"]`
- No missing parameters detected

#### Solver Configuration
- **86/86 successful** - All configurations can be used to configure solver
  - 2D: 28/28 successful
  - 3D: 58/58 successful
- All parameters are correctly passed to `solver.set_settings()`

#### Problem Type Statistics
- **80/86** are transient problems (require time stepping)
  - 2D: 27/28 transient
  - 3D: 53/58 transient
- **58/86** have contact enabled
  - 2D: 20/28 have contact
  - 3D: 38/58 have contact
- **6/86** are static problems

### common.json Support

✅ **common.json Automatic Merging**: API fully supports `common.json` reference and merging functionality.

#### Merging Rules

1. **Automatic Detection**: If configuration file contains `"common": "path"` reference, it will be automatically loaded and merged
2. **Deep Merging**: Nested dictionaries are recursively merged (e.g., `output.paraview` will merge `file_name` and `options`)
3. **Priority**: Original configuration values override default values in `common.json`
4. **Automatic Removal**: `common` key is automatically removed after merging

#### Example

```json
// common.json
{
  "contact": {"enabled": true, "dhat": 0.001},
  "output": {
    "paraview": {
      "options": {"material": true}
    }
  }
}

// 5-squares.json
{
  "common": "../../common.json",
  "output": {
    "paraview": {
      "file_name": "5-squares.pvd"
    }
  }
}

// After merging
{
  "contact": {"enabled": true, "dhat": 0.001},
  "output": {
    "paraview": {
      "file_name": "5-squares.pvd",  // From original configuration
      "options": {"material": true}   // From common.json
    }
  }
}
```

**Test Results**: Among 86 example files, all files referencing `common.json` can be correctly merged.

### Loading Existing Examples

All examples can be loaded using the following methods:

```python
from polyfempy.api import SimulationConfig, solve

# Method 1: Load from file path
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")

# Method 2: Load from dictionary
import json
with open("data/contact/examples/2D/unit-tests/5-squares.json") as f:
    config_dict = json.load(f)
cfg = SimulationConfig.from_json_dict(config_dict)

# Method 3: Direct solve (for static problems)
result = solve(cfg="data/contact/examples/2D/unit-tests/5-squares.json")
```

### Supported Parameters (Verified)

Through testing 86 real examples, confirmed that API supports all parameters in PolyFEM JSON format:

- ✅ **geometry**: Mesh files, transformations, volume_selection, surface_selection
- ✅ **materials**: All material types (LinearElasticity, NeoHookean, etc.) and parameters like E, nu, rho
- ✅ **boundary_conditions**: Dirichlet, Neumann, pressure, RHS, etc.
- ✅ **time**: Transient settings (t0, tend, dt, integrator)
- ✅ **contact**: Contact settings (enabled, dhat, mu, epsv, etc.)
- ✅ **solver**: Linear/nonlinear solver settings
- ✅ **output**: Paraview, JSON output settings
- ✅ **space**: Discretization order (supports list format, e.g., `[{"id": 2, "order": 2}]`)
- ✅ **common**: JSON references (automatic merging, supports deep nested merging)

### Transient Problem Handling

**Important Note**: Most examples (80/86) are transient problems requiring time stepping. The current `solve()` function can automatically handle static problems, but for transient problems, the low-level API needs to be used:

```python
import polyfempy as pf
import json
from polyfempy.api import SimulationConfig

# Load configuration
cfg = SimulationConfig.from_json_file("data/contact/examples/2D/unit-tests/5-squares.json")
full_json = cfg.extras["_full_json_config"]

# Configure solver
solver = pf.Solver()
solver.set_settings(json.dumps(full_json), strict_validation=False)
solver.load_mesh_from_settings()
solver.build_basis()
solver.assemble()

# Time stepping
config = solver.settings()
t0 = config["time"]["t0"]
dt = config["time"]["dt"]
sol = solver.init_timestepping(t0, dt)

# Run time steps
for i in range(1, 5):
    for t in range(1):
        sol = solver.step_in_time(sol, t0, dt, t+1)
    t0 += dt
    solver.export_vtu(f"step_{i}.vtu", sol, np.zeros((0, 0)), t0, dt)
```

### Future Enhancements

To fully support transient problems in the high-level `solve()` API, consider:

1. **Automatic transient detection**: Check `time` configuration
2. **Time stepping loop**: Automatically handle `init_timestepping()` and `step_in_time()`
3. **Result aggregation**: Return time series results for transient problems

---

## Adding New Parameters

### Current Design

Uses `_EXTRAS_PROMOTION_RULES` dictionary to configure which parameters need to be promoted and how to validate and convert them.

### How to Add New Parameters

#### Step 1: Define Validator Function (if needed)

```python
def _validate_positive_float(v):
    """Validate and convert to positive float."""
    v = float(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v
```

#### Step 2: Add Configuration to `_EXTRAS_PROMOTION_RULES`

Find `_EXTRAS_PROMOTION_RULES` dictionary in `polyfempy/api/config.py`, add new parameter:

```python
_EXTRAS_PROMOTION_RULES = {
    "max_iters": (
        _validate_positive_int,
        "extras['max_iters'] must be a positive integer, got {value!r} (type: {type_name})"
    ),
    "random_seed": (
        _validate_int_or_none,
        "extras['random_seed'] must be an integer or None, got {value!r} (type: {type_name})"
    ),
    # Add new parameter: just this line!
    "tolerance": (
        _validate_positive_float,
        "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
    ),
}
```

#### Step 3: Done

No need to modify `to_dict()` method, the system will handle it automatically.

### Common Validator Patterns

#### Pattern 1: Positive Integer
```python
def _validate_positive_int(v):
    v = int(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v
```

#### Pattern 2: Positive Float
```python
def _validate_positive_float(v):
    v = float(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v
```

#### Pattern 3: Optional Integer (allows None)
```python
def _validate_int_or_none(v):
    if v is None:
        return None
    return int(v)
```

#### Pattern 4: Enumeration Value (string)
```python
def _validate_solver_type(v):
    v = str(v)
    if v not in ["linear", "nonlinear", "mixed"]:
        raise ValueError("must be 'linear', 'nonlinear', or 'mixed'")
    return v
```

#### Pattern 5: Range Check
```python
def _validate_probability(v):
    v = float(v)
    if not (0.0 <= v <= 1.0):
        raise ValueError("must be between 0 and 1")
    return v
```

### Complete Example

Suppose you want to add `tolerance` parameter:

```python
# 1. Define validator (before _EXTRAS_PROMOTION_RULES)
def _validate_positive_float(v):
    v = float(v)
    if v <= 0:
        raise ValueError("must be positive")
    return v

# 2. Add to _EXTRAS_PROMOTION_RULES
_EXTRAS_PROMOTION_RULES = {
    # ... existing parameters ...
    "tolerance": (
        _validate_positive_float,
        "extras['tolerance'] must be a positive float, got {value!r} (type: {type_name})"
    ),
}

# 3. Use
cfg = SimulationConfig(extras={"tolerance": "1e-6"})
d = cfg.to_dict()
# d["tolerance"] = 1e-6 (automatically converted to float)
```

### Advantages

- ✅ **Extensibility**: Adding new parameters only requires adding one line to the dictionary
- ✅ **Consistency**: All parameters use the same validation and promotion mechanism
- ✅ **Maintainability**: All parameter configurations are centralized in one place
- ✅ **Type Safety**: Automatic type conversion (string → number)

---

## Summary

### Data Flow

```
User Input (extras) 
  → SimulationConfig.extras 
  → to_dict() (validate, convert, promote to top level) 
  → settings dictionary 
  → Used by solve() / C++ extension
```

### Key Points

- **`extras`**: User-input extra parameters (flexibility)
- **`to_dict()`**: Responsible for validation, conversion, and promotion (compatibility)
- **`settings`**: Format used by the C++ extension (simplicity)

### Validation Status

- ✅ **Validated and converted**: `max_iters`, `random_seed`
- ⚠️ **Not validated but supported**: All other JSON parameters (via full JSON mode)
- ❌ **Current limitation**: Parameters input via `extras` (except common parameters) will not be validated

### Usage Recommendations

1. **Simple configuration**: Use basic fields + common parameters in `extras`
2. **Complete configuration**: Load from JSON file
3. **Need extra parameters**: Use full JSON dictionary

---

