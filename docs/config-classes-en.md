# Config Classes Guide

This document provides detailed information about the various configuration classes used in `SimulationConfig`, including Material, BoundaryConditions, and ProblemParams.

---

## Table of Contents

1. [Why Use Classes?](#why-use-classes)
2. [Available Classes](#available-classes)
3. [Convenience Methods](#convenience-methods)
4. [Convenience Factories](#convenience-factories)
5. [Backward Compatibility](#backward-compatibility)
6. [Complete Examples](#complete-examples)

---

## Why Use Classes?

The main reason for using classes instead of dictionaries is **to provide better IDE autocomplete support**.

### Problem: Limitations of Dict + Typing

```python
# ❌ Using dict - IDE cannot autocomplete
cfg = SimulationConfig(materials={"E": 2100, "nu": 0.3})
cfg.materials["E"] = 2100  # IDE doesn't know "E" is a valid key
cfg.materials["wrong_key"] = 100  # IDE won't report an error
cfg.materials[""]  # After typing quotes, IDE won't suggest available keys
```

**Problems**:
- IDE cannot know which keys are valid in the `materials` dictionary
- When users type `cfg.materials["`, IDE won't suggest `"E"` or `"nu"`
- Spelling errors won't be caught by IDE
- Parameters for different problem types are easily confused

### Solution: Use Classes

```python
# ✅ Using class - IDE autocomplete
from polyfempy.api.config import Material, GravityParams

material = Material(E=2100, nu=0.3)
cfg = SimulationConfig(materials=material)
material.E  # IDE will suggest this is a valid attribute
material.nu  # IDE will suggest this is a valid attribute

params = GravityParams(force=0.1)
cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
params.force  # IDE will suggest this is a valid attribute
```

**Advantages**:
- ✅ IDE can provide complete autocomplete
- ✅ Stricter type checking
- ✅ Clearer code, less error-prone
- ✅ Can add validation logic
- ✅ Prevents confusion between parameters for different problem types

---

## Available Classes

### Material Classes

PolyFEM provides various material classes, each with dedicated class definitions that support strong IDE parameter hints and multiple input modes.

#### Generic Material Class

The basic `Material` class for simple linear elastic materials:

```python
from polyfempy.api.config import Material

material = Material(
    E=2100,              # Young's modulus
    nu=0.3,              # Poisson's ratio
    rho=1.0,             # Density
    type="LinearElasticity"  # Material type
)

# IDE will autocomplete the following attributes:
# material.E
# material.nu
# material.rho
# material.type
```

**Note**: While the `Material` class supports a `type` parameter to specify material types, it is **recommended to use dedicated material classes** (such as `NeoHookean`, `LinearElasticity`, etc.) for better type hints and parameter validation.

#### Dedicated Material Classes

The following material classes provide stronger type safety and IDE support:

##### NeoHookean

NeoHookean material, supports two input modes:

```python
from polyfempy.api.config import NeoHookean

# Mode 1: E-nu input (Young's modulus and Poisson's ratio)
material = NeoHookean(
    E=2100,      # Young's modulus (required)
    nu=0.3,      # Poisson's ratio (required)
    id=0,        # Material ID (optional, default 0)
    rho=1,       # Density (optional, default 1)
    phi=0,       # First angle (optional, default 0)
    psi=0        # Second angle (optional, default 0)
)

# Mode 2: lambda-mu input (Lamé parameters)
material = NeoHookean(
    lambda_=1000,  # First Lamé parameter (required)
    mu=800,        # Shear modulus (required)
    id=0,
    rho=1,
    phi=0,
    psi=0
)
```

##### IsochoricNeoHookean

Isochoric NeoHookean material, also supports E-nu and lambda-mu inputs:

```python
from polyfempy.api.config import IsochoricNeoHookean

# E-nu input
material = IsochoricNeoHookean(E=2100, nu=0.3)

# lambda-mu input
material = IsochoricNeoHookean(lambda_=1000, mu=800)
```

##### LinearElasticity

Linear elasticity material, supports two input modes:

```python
from polyfempy.api.config import LinearElasticity

# Mode 1: E-nu input
material = LinearElasticity(
    E=2100,
    nu=0.3,
    id=0,
    rho=1,
    phi=0,  # Supported in E-nu mode
    psi=0   # Supported in E-nu mode
)

# Mode 2: lambda-mu input
material = LinearElasticity(
    lambda_=1000,
    mu=800,
    id=0,
    rho=1
)
```

##### HookeLinearElasticity

Hooke linear elasticity material, supports two input modes:

```python
from polyfempy.api.config import HookeLinearElasticity

# Mode 1: E-nu input
material = HookeLinearElasticity(
    E=2100,
    nu=0.3,
    id=0,
    rho=1,
    fiber_direction=[0, 0, 0]  # Fiber direction vector
)

# Mode 2: elasticity_tensor input
material = HookeLinearElasticity(
    elasticity_tensor=[...],  # Full elasticity tensor
    id=0,
    rho=1,
    fiber_direction=[0, 0, 0]
)
```

##### SaintVenant

Saint-Venant material, supports two input modes:

```python
from polyfempy.api.config import SaintVenant

# Mode 1: E-nu input
material = SaintVenant(
    E=2100,
    nu=0.3,
    id=0,
    rho=1,
    phi=0,
    psi=0,
    fiber_direction=[0, 0, 0]
)

# Mode 2: elasticity_tensor input
material = SaintVenant(
    elasticity_tensor=[...],
    id=0,
    rho=1,
    phi=0,
    psi=0,
    fiber_direction=[0, 0, 0]
)
```

##### MooneyRivlin

Mooney-Rivlin material:

```python
from polyfempy.api.config import MooneyRivlin

material = MooneyRivlin(
    c1=0.5,      # First Mooney-Rivlin parameter (required)
    c2=0.1,      # Second Mooney-Rivlin parameter (required)
    k=1000,      # Bulk modulus (required)
    id=0,        # Material ID (optional, default 0)
    rho=1        # Density (optional, default 1)
)
```

##### MooneyRivlin3Param

Three-parameter Mooney-Rivlin material:

```python
from polyfempy.api.config import MooneyRivlin3Param

material = MooneyRivlin3Param(
    c1=0.5,      # First Mooney-Rivlin parameter (required)
    c2=0.1,      # Second Mooney-Rivlin parameter (required)
    c3=0.05,     # Third Mooney-Rivlin parameter (required)
    d1=1000,     # First volumetric parameter (required)
    id=0,
    rho=1
)
```

##### MooneyRivlin3ParamSymbolic

Symbolic three-parameter Mooney-Rivlin material:

```python
from polyfempy.api.config import MooneyRivlin3ParamSymbolic

material = MooneyRivlin3ParamSymbolic(
    c1=0.5,
    c2=0.1,
    c3=0.05,
    d1=1000,
    id=0,
    rho=1
)
```

##### UnconstrainedOgden

Unconstrained Ogden material:

```python
from polyfempy.api.config import UnconstrainedOgden

material = UnconstrainedOgden(
    alphas=2.0,           # Alpha parameters (required)
    mus=[1.0, 0.5],       # Mu parameters list (required)
    Ds=[0.1, 0.2],        # D parameters list (required)
    id=0,
    rho=1
)
```

##### IncompressibleOgden

Incompressible Ogden material:

```python
from polyfempy.api.config import IncompressibleOgden

material = IncompressibleOgden(
    c=1.0,        # C parameters (required, can be float/string/object/list)
    m=2.0,        # M parameters (required, can be float/string/object/list)
    k=1000,       # Bulk modulus (required)
    id=0,
    rho=1
)
```

##### IncompressibleLinearElasticity

Incompressible linear elasticity material:

```python
from polyfempy.api.config import IncompressibleLinearElasticity

material = IncompressibleLinearElasticity(
    E=2100,       # Young's modulus (required)
    nu=0.3,       # Poisson's ratio (required)
    id=0,
    rho=1
)
```

##### Stokes

Stokes fluid material:

```python
from polyfempy.api.config import Stokes

material = Stokes(
    viscosity=0.1,  # Viscosity (required)
    id=0,
    rho=1
)
```

##### NavierStokes

Navier-Stokes fluid material:

```python
from polyfempy.api.config import NavierStokes

material = NavierStokes(
    viscosity=0.1,  # Viscosity (required)
    id=0,
    rho=1
)
```

##### OperatorSplitting

Operator splitting material:

```python
from polyfempy.api.config import OperatorSplitting

material = OperatorSplitting(
    viscosity=0.1,  # Viscosity (required)
    id=0,
    rho=1
)
```

##### Electrostatics

Electrostatics material:

```python
from polyfempy.api.config import Electrostatics

material = Electrostatics(
    epsilon=8.85e-12,  # Permittivity (required)
    id=0,
    rho=1
)
```

#### Usage Examples

```python
from polyfempy.api.config import (
    SimulationConfig,
    NeoHookean,
    LinearElasticity,
    Stokes
)

# Using dedicated material classes (recommended)
neo_hookean = NeoHookean(E=2100, nu=0.3)
cfg1 = SimulationConfig(materials=neo_hookean)

linear_elastic = LinearElasticity(E=2100, nu=0.3)
cfg2 = SimulationConfig(materials=linear_elastic)

stokes = Stokes(viscosity=0.1)
cfg3 = SimulationConfig(materials=stokes)

# All material classes support IDE autocomplete
# After typing material., IDE will suggest all available attributes
```

### BoundaryConditions

Boundary condition container class with IDE autocomplete support.

```python
from polyfempy.api.config import BoundaryConditions

bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])  # IDE will suggest parameters
bc.add_neumann(id=2, value=[0.0, -1000.0])  # IDE will suggest parameters
bc.set_rhs([1.0, 0.0])  # IDE will suggest parameters

# IDE will autocomplete the following attributes:
# bc.dirichlet_boundary
# bc.neumann_boundary
# bc.rhs

cfg = SimulationConfig(boundary_conditions=bc)
```

### ProblemParams Classes

Parameter classes for different problem types:

#### GravityParams

```python
from polyfempy.api.config import GravityParams

params = GravityParams(force=0.1)  # IDE will suggest 'force'
cfg = SimulationConfig(problem_type="Gravity", problem_params=params)
```

#### TorsionParams

```python
from polyfempy.api.config import TorsionParams

params = TorsionParams(
    axis_coordinate=2,  # IDE will suggest all parameters
    n_turns=0.5,
    fixed_boundary=5,
    turning_boundary=6
)
cfg = SimulationConfig(problem_type="TorsionElastic", problem_params=params)
```

#### FlowParams

```python
from polyfempy.api.config import FlowParams

params = FlowParams(
    inflow=1,           # IDE will suggest all parameters
    outflow=3,
    inflow_amount=0.25,  # Note: correct spelling in class (not inflow_amout)
    outflow_amount=0.25,
    direction=0,
    obstacle=[7]
)
cfg = SimulationConfig(problem_type="Flow", problem_params=params)
```

#### FlowWithObstacleParams

```python
from polyfempy.api.config import FlowWithObstacleParams

params = FlowWithObstacleParams(U=1.5, time_dependent=True)
cfg = SimulationConfig(problem_type="FlowWithObstacle", problem_params=params)
```

---

## Convenience Methods

`SimulationConfig` provides convenience methods to set parameters (all support IDE autocomplete):

```python
cfg = SimulationConfig()

# Set material parameters
cfg.set_material(E=2100, nu=0.3)  # IDE will suggest all parameters

# Set boundary conditions
cfg.set_dirichlet_boundary(id=4, value=[0.0, 0.0])  # IDE will suggest
cfg.set_neumann_boundary(id=2, value=[0.0, -1000.0])  # IDE will suggest
cfg.set_rhs([1.0, 0.0])  # IDE will suggest

# All methods support method chaining
cfg.set_material(E=2100, nu=0.3) \
   .set_dirichlet_boundary(id=4, value=[0.0, 0.0]) \
   .set_neumann_boundary(id=2, value=[0.0, -1000.0])
```

---

## Convenience Factories

All convenience factory methods now automatically use classes:

```python
# Gravity - automatically uses GravityParams
cfg = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
cfg.problem_params.force  # ✅ IDE will suggest

# Torsion - automatically uses TorsionParams
cfg = SimulationConfig.torsion(axis_coordinate=2, n_turns=0.5)
cfg.problem_params.axis_coordinate  # ✅ IDE will suggest

# Flow - automatically uses FlowParams
cfg = SimulationConfig.flow(inflow=1, outflow=3, inflow_amount=0.25)
cfg.problem_params.inflow_amount  # ✅ Correct spelling!
```

---

## Backward Compatibility

All existing code continues to work:

1. **Dict input still supported**: `SimulationConfig(materials={"E": 2100})` still works
2. **Automatic conversion**: Dict input is automatically converted to the corresponding class (if needed)
3. **to_dict() method**: All classes have a `to_dict()` method for backend compatibility

---

## Complete Examples

### Example 1: Using Generic Material Class

```python
from polyfempy.api.config import (
    SimulationConfig,
    Material,
    BoundaryConditions,
    GravityParams
)

# Method 1: Using classes (recommended)
material = Material(E=2100, nu=0.3)
bc = BoundaryConditions()
bc.add_dirichlet(id=4, value=[0.0, 0.0])
gravity_params = GravityParams(force=0.1)

cfg = SimulationConfig(
    pde="LinearElasticity",
    discr_order=1,
    materials=material,
    boundary_conditions=bc,
    problem_type="Gravity",
    problem_params=gravity_params
)
```

### Example 2: Using Dedicated Material Classes (Recommended)

```python
from polyfempy.api.config import (
    SimulationConfig,
    NeoHookean,
    LinearElasticity,
    Stokes,
    BoundaryConditions
)

# NeoHookean material (E-nu input)
neo_hookean = NeoHookean(E=2100, nu=0.3, rho=1.0)
cfg1 = SimulationConfig(materials=neo_hookean)

# NeoHookean material (lambda-mu input)
neo_hookean_lame = NeoHookean(lambda_=1000, mu=800)
cfg2 = SimulationConfig(materials=neo_hookean_lame)

# LinearElasticity material
linear_elastic = LinearElasticity(E=2100, nu=0.3)
cfg3 = SimulationConfig(materials=linear_elastic)

# Stokes fluid material
stokes = Stokes(viscosity=0.1, rho=1.0)
cfg4 = SimulationConfig(materials=stokes)

# All material classes support IDE autocomplete
# After typing material., IDE will suggest all available attributes
```

### Example 3: Using Convenience Methods

```python
cfg = SimulationConfig()
cfg.set_material(E=2100, nu=0.3) \
   .set_dirichlet_boundary(id=4, value=[0.0, 0.0]) \
   .set_rhs([1.0, 0.0])
```

### Example 4: Using Convenience Factories

```python
cfg = SimulationConfig.gravity(force=0.1, E=1e6, nu=0.3)
```

### Example 5: Backward Compatible (Still Supported)

```python
cfg = SimulationConfig(
    materials={"E": 2100, "nu": 0.3},  # ⚠️ IDE cannot suggest
    boundary_conditions={"dirichlet_boundary": [...]},  # ⚠️ IDE cannot suggest
    problem_params={"force": 0.1}  # ⚠️ IDE cannot suggest
)
```

### Example 6: Complex Material Configurations

```python
from polyfempy.api.config import (
    SimulationConfig,
    MooneyRivlin,
    UnconstrainedOgden,
    HookeLinearElasticity
)

# MooneyRivlin material
mooney_rivlin = MooneyRivlin(c1=0.5, c2=0.1, k=1000, rho=1.0)
cfg1 = SimulationConfig(materials=mooney_rivlin)

# UnconstrainedOgden material
ogden = UnconstrainedOgden(
    alphas=2.0,
    mus=[1.0, 0.5],
    Ds=[0.1, 0.2],
    rho=1.0
)
cfg2 = SimulationConfig(materials=ogden)

# HookeLinearElasticity material (elasticity_tensor input)
hooke = HookeLinearElasticity(
    elasticity_tensor=[...],  # Full elasticity tensor
    fiber_direction=[1, 0, 0]
)
cfg3 = SimulationConfig(materials=hooke)
```

