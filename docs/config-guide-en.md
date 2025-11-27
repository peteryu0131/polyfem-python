# SimulationConfig Configuration Guide

This document provides a comprehensive guide on using `SimulationConfig`, including parameter input, validation, data flow, and extension.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Flow Process](#data-flow-process)
3. [Parameter Input Methods](#parameter-input-methods)
4. [Parameter Validation](#parameter-validation)
5. [JSON Parameter Support](#json-parameter-support)
6. [Real-World Examples](#real-world-examples)
7. [Adding New Parameters](#adding-new-parameters)

---

## Quick Start

### Basic Usage

```python
from polyfempy.api import SimulationConfig, solve
import numpy as np

# Create simple configuration
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}
)

# Use configuration
V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
C = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
result = solve(V, C, cfg)
```

### Load from JSON File

```python
# Load full configuration from JSON file
cfg = SimulationConfig.from_json_file("config.json")
result = solve(vertices=None, cells=None, cfg=cfg)
```

---

## Data Flow Process

### Why So Many Dictionaries?

There are several different dictionaries used at different stages in the API:

1. **`SimulationConfig.extras`** - User-inputted extra parameters
2. **Dictionary returned by `SimulationConfig.to_dict()`** - Configuration passed to backend
3. **`settings` dictionary (passed to `solve_impl()`)** - Configuration actually used by backend

### Data Flow Process

#### Stage 1: User Creates Configuration

```python
# User creates SimulationConfig
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}  # ← Put in extras
)
```

**Why put in `extras`?**
- `SimulationConfig` is a dataclass with fixed fields: `pde`, `discr_order`, `materials`, `boundary_conditions`, `extras`, `selection`, `problem_type`, `problem_params`
- `max_iters` and `random_seed` are not direct fields of `SimulationConfig`
- So they need to be put in the `extras` "miscellaneous" dictionary

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
- Backends (`backend_dummy.py`, `backend_nanobind.py`) expect `max_iters` and `random_seed` at the dictionary's top level
- Backend code: `max_iters = settings.get("max_iters", 10)` - Read directly from top level
- If only in `extras`, backend would need `settings["extras"]["max_iters"]`, which is inconvenient

#### Stage 3: Pass to Backend

```python
# solve() function calls backend
from .backend_dummy import solve_impl

settings = cfg.to_dict()  # Get dictionary
result = solve_impl(V, C, settings, callbacks=None)
```

**How backend uses it:**

```python
# backend_dummy.py
def solve_impl(V, C, settings, callbacks):
    # Read directly from top level
    max_iters = settings.get("max_iters", 10)      # ← Read from top level
    random_seed = settings.get("random_seed", 42)  # ← Read from top level
    
    # Use these parameters to run solve
    for i in range(max_iters):
        # ... solve logic ...
```

### Complete Data Flow Example

```python
# 1. User creates configuration
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={"max_iters": 10, "random_seed": 42}
)

# 2. Convert to dictionary (called internally)
settings = cfg.to_dict()
# settings = {
#     "pde": "linear_elasticity",
#     "discr_order": 1,
#     "materials": {"E": 1e6, "nu": 0.3},
#     "boundary_conditions": {},
#     "extras": {"max_iters": 10, "random_seed": 42},  # Original extras preserved
#     "max_iters": 10,        # Promoted to top level for easy backend access
#     "random_seed": 42       # Promoted to top level for easy backend access
# }

# 3. solve() function calls backend
result = solve(V, C, cfg)  # Internally calls cfg.to_dict() to get settings

# 4. Backend uses settings
# In backend_dummy.py:
# max_iters = settings.get("max_iters", 10)  # Read from top level, gets 10
```

### Design Reasons

**Design Reason 1: Type Safety vs Flexibility**
- **`SimulationConfig` is type-safe**: Has fixed fields, IDE can auto-complete
- **`extras` provides flexibility**: Can store arbitrary extra parameters without modifying class definition
- **`to_dict()` provides conversion**: Converts type-safe object to flexible dictionary

**Design Reason 2: Backend Compatibility**
- **Backends need simple dictionaries**: Backend code expects simple `dict`, not complex objects
- **Promote common parameters**: `max_iters` and `random_seed` are common parameters, promoting to top level is convenient
- **Preserve original data**: `extras` still preserved in case other extra parameters need to be accessed

**Design Reason 3: Support Full JSON Configuration**
```python
# If loading from full JSON
cfg = SimulationConfig.from_json_file("config.json")
# Full JSON stored in extras["_full_json_config"]
# to_dict() will directly return full JSON instead of constructed dictionary
```

---

## Parameter Input Methods

### Method 1: Basic Fields (Recommended for Simple Configurations)

```python
cfg = SimulationConfig(
    pde="linear_elasticity",
    discr_order=2,
    materials={"E": 1e6, "nu": 0.3},
    boundary_conditions={
        "dirichlet_boundary": [{"id": 4, "value": [0.0, 0.0]}],
        "rhs": [1.0, 0.0]
    }
)
```

### Method 2: Input Extra Parameters via extras

```python
cfg = SimulationConfig(
    pde="linear_elasticity",
    materials={"E": 1e6, "nu": 0.3},
    extras={
        # Common parameters (will be validated and promoted to top level)
        "max_iters": 10,
        "random_seed": 42,
        
        # Other JSON parameters (remain in extras)
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

### Method 3: Load from JSON File (Recommended for Full Configurations)

```python
# Load full configuration from JSON file
cfg = SimulationConfig.from_json_file("config.json")
# All parameters saved in extras["_full_json_config"]
```

### Method 4: Load from Dictionary

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

## Parameter Validation

### Validation Levels

#### 1. SimulationConfig Level Validation

**Parameters Checked**:
- `discr_order`: Must be positive integer (via `validate()` method)
- `materials['E']` and `materials['nu']`: Must be numbers (via `validate()` method)

**Example**:
```python
# ✅ Will be checked
cfg = SimulationConfig(discr_order=-1)  # ValueError: discr_order must be positive
cfg = SimulationConfig(materials={"E": "abc"})  # ValueError: materials['E'] must be a number
```

#### 2. to_dict() Level Validation

**Current Behavior**:
- `to_dict()` **validates and converts** common parameters in `extras`
- `max_iters`: Must be positive integer, string `"10"` auto-converted to integer `10`
- `random_seed`: Must be integer or None, string `"42"` auto-converted to integer `42`
- Invalid values immediately raise `ValueError` with clear error message

**Example**:
```python
# ✅ String auto-converted to integer
cfg = SimulationConfig(extras={"max_iters": "10"})
d = cfg.to_dict()
print(d["max_iters"])  # 10 (integer)
print(type(d["max_iters"]))  # <class 'int'>

# ❌ Invalid string immediately raises error
cfg = SimulationConfig(extras={"max_iters": "abc"})
d = cfg.to_dict()  # ValueError: extras['max_iters'] must be a positive integer, got 'abc'

# ❌ Negative number immediately raises error
cfg = SimulationConfig(extras={"max_iters": -5})
d = cfg.to_dict()  # ValueError: extras['max_iters'] must be a positive integer, got -5
```

### Validation Rules

1. **`max_iters`**:
   - Must be positive integer
   - String `"10"` auto-converted to integer `10`
   - Negative numbers or invalid strings raise `ValueError`

2. **`random_seed`**:
   - Must be integer or `None`
   - String `"42"` auto-converted to integer `42`
   - Invalid strings raise `ValueError`

3. **Other JSON Parameters** (`solver`, `time`, `output`, `contact`, `geometry`, etc.):
   - Can be input via `extras`
   - Not validated (no type checking)
   - Remain in `extras`, not promoted to top level
   - If using full JSON mode (load from file), all parameters are preserved
   - Require backend (C++ bindings) support to use

### Unknown Parameter Handling

**Design Behavior**:
- Unknown parameters are not validated
- Remain in `extras`, but not promoted to top level
- No error raised (allows custom parameters)

**Example**:
```python
# User made a typo
cfg = SimulationConfig(extras={"max_iter": 10})  # Should be max_iters

# No error, but parameter ignored (uses default)
settings = cfg.to_dict()
# settings["max_iters"] doesn't exist, uses default value 10
```

**Why This Design?**
- Allows users to add custom parameters to `extras` that may be used by other tools
- Only validate known critical parameters (`max_iters`, `random_seed`)

---

## JSON Parameter Support

### Complete PolyFEM JSON Configuration Parameter List

According to `from_json_dict()` documentation, PolyFEM JSON configuration supports all these parameters:

#### 1. Basic Parameters
- `pde`: PDE type ("Poisson", "LinearElasticity", "NonLinearElasticity", etc.)
- `discr_order`: Discretization order (1, 2, ...)
- `space`: Space configuration (contains `discr_order`, etc.)

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

**Method 1: Load from JSON File (Recommended)**
```python
cfg = SimulationConfig.from_json_file("config.json")
# All parameters preserved in full JSON
```

**Method 2: Input via extras**
```python
cfg = SimulationConfig(extras={
    "solver": {...},
    "time": {...},
    "output": {...}
})
# Parameters remain in extras, not validated
```

**Method 3: Input via Dictionary**
```python
config_dict = {"solver": {...}, "time": {...}}
cfg = SimulationConfig.from_json_dict(config_dict)
# Converted to full JSON configuration
```

### to_dict() Behavior

1. **If Full JSON Configuration Exists** (`extras["_full_json_config"]`):
   ```python
   # Directly return full JSON, contains all parameters
   return dict(self.extras["_full_json_config"])
   ```

2. **If No Full JSON Configuration**:
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
   
   # Other extras parameters remain in extras
   result["extras"] = dict(extras)
   ```

---

## Real-World Examples

### Compatibility with polyfem-data Examples

✅ **API Design Success**: All 86 JSON configuration files from `polyfem-data/contact/examples/2D` and `3D` can be successfully loaded and configured using the new API.
- **2D Examples**: 28/28 successful
- **3D Examples**: 58/58 successful

### Test Results

#### Configuration Loading
- **86/86 successful** - All JSON files can be loaded into `SimulationConfig`
  - 2D: 28/28 successful
  - 3D: 58/58 successful
- All parameters saved in `extras["_full_json_config"]`
- No missing parameters detected

#### Solver Configuration
- **86/86 successful** - All configurations can be used to configure solver
  - 2D: 28/28 successful
  - 3D: 58/58 successful
- All parameters correctly passed to `solver.set_settings()`

#### Problem Type Statistics
- **80/86** are transient problems (require time stepping)
  - 2D: 27/28 transient
  - 3D: 53/58 transient
- **58/86** have contact enabled
  - 2D: 20/28 with contact
  - 3D: 38/58 with contact
- **6/86** are static problems

### common.json Support

✅ **common.json Auto-Merge**: API fully supports `common.json` reference and merge functionality.

#### Merge Rules

1. **Auto-Detection**: If config file contains `"common": "path"` reference, automatically load and merge
2. **Deep Merge**: Nested dictionaries recursively merged (e.g., `output.paraview` merges `file_name` and `options`)
3. **Priority**: Original config values override defaults in `common.json`
4. **Auto-Removal**: `common` key automatically removed after merge

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

// After merge
{
  "contact": {"enabled": true, "dhat": 0.001},
  "output": {
    "paraview": {
      "file_name": "5-squares.pvd",  // From original config
      "options": {"material": true}   // From common.json
    }
  }
}
```

**Test Result**: All 86 example files that reference `common.json` merge correctly.

### Loading Existing Examples

All examples can be loaded using:

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

Through testing 86 real examples, confirmed API supports all parameters in PolyFEM JSON format:

- ✅ **geometry**: Mesh files, transformations, volume_selection, surface_selection
- ✅ **materials**: All material types (LinearElasticity, NeoHookean, etc.) and E, nu, rho parameters
- ✅ **boundary_conditions**: Dirichlet, Neumann, pressure, RHS, etc.
- ✅ **time**: Transient settings (t0, tend, dt, integrator)
- ✅ **contact**: Contact settings (enabled, dhat, mu, epsv, etc.)
- ✅ **solver**: Linear/nonlinear solver settings
- ✅ **output**: Paraview, JSON output settings
- ✅ **space**: Discretization order (supports list format, e.g., `[{"id": 2, "order": 2}]`)
- ✅ **common**: JSON references (auto-merge, supports deep nested merging)

### Transient Problem Handling

**Important Note**: Most examples (80/86) are transient problems requiring time stepping. Current `solve()` function can automatically handle static problems, but for transient problems, need to use low-level API:

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

To fully support transient problems in high-level `solve()` API, consider:

1. **Auto-detect transient problems**: Check `time` configuration
2. **Time stepping loop**: Auto-handle `init_timestepping()` and `step_in_time()`
3. **Result aggregation**: Return time series results for transient problems

---

## Adding New Parameters

### Current Design

Use `_EXTRAS_PROMOTION_RULES` dictionary to configure which parameters need promotion and how to validate/convert them.

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

No need to modify `to_dict()` method, system handles it automatically.

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

#### Pattern 3: Optional Integer (Allow None)
```python
def _validate_int_or_none(v):
    if v is None:
        return None
    return int(v)
```

#### Pattern 4: Enum Value (String)
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
# d["tolerance"] = 1e-6 (auto-converted to float)
```

### Advantages

- ✅ **Extensibility**: Adding new parameter only requires one line in dictionary
- ✅ **Consistency**: All parameters use same validation and promotion mechanism
- ✅ **Maintainability**: All parameter configuration centralized in one place
- ✅ **Type Safety**: Automatic type conversion (string → number)

---

## Summary

### Data Flow

```
User Input (extras) 
  → SimulationConfig.extras 
  → to_dict() (validate, convert, promote to top level) 
  → settings dictionary 
  → Backend Use
```

### Key Points

- **`extras`**: User-inputted extra parameters (flexibility)
- **`to_dict()`**: Responsible for validation, conversion, and promotion (compatibility)
- **`settings`**: Format used by backend (simplicity)

### Validation Status

- ✅ **Validated and Converted**: `max_iters`, `random_seed`
- ⚠️ **Not Validated But Supported**: All other JSON parameters (via full JSON mode)
- ❌ **Current Limitation**: Parameters input via `extras` (except common parameters) are not validated

### Usage Recommendations

1. **Simple Configuration**: Use basic fields + common parameters in `extras`
2. **Full Configuration**: Load from JSON file
3. **Need Extra Parameters**: Use full JSON dictionary

---

## Related Documentation

- [API Architecture Document](api-architecture-en.md) - Complete API architecture explanation
- [Example Code](../polyfempy/api/examples/README.md) - Real-world usage examples

