# PolyFEM API Examples

This directory contains runnable examples demonstrating how to use the PolyFEM Python API.

## Examples

### 1. `run_dummy_elasticity.py`
**Basic Dummy backend example**

Demonstrates the simplest use case:
- Creating a simple 2D mesh (unit square with 2 triangles)
- Configuring simulation parameters
- Running the solver with Dummy backend
- Displaying results and metadata

**Run:**
```bash
python -m polyfempy.api.examples.run_dummy_elasticity
```

---

### 2. `run_elasticity.py`
**Minimal 2D Linear Elasticity**

Shows how to:
- Set up a linear elasticity problem
- Configure materials and boundary conditions
- Export results to VTU format

**Run:**
```bash
python -m polyfempy.api.examples.run_elasticity
```

---

### 3. `load_from_json.py`
**Load configuration from JSON file**

Demonstrates how to:
- Load a complete simulation configuration from a JSON file
- Use JSON files that include geometry (mesh will be loaded automatically)
- Work with full PolyFEM JSON configurations

**Run:**
```bash
python -m polyfempy.api.examples.load_from_json
```

**Note:** Update the `json_file` path in the script to point to your JSON configuration file.

---

### 4. `parameter_sweep.py`
**Parameter sweep - study parameter sensitivity**

Shows how to:
- Run multiple simulations with different parameters
- Compare results across different configurations
- Study the effect of material properties (e.g., Young's modulus)

**Run:**
```bash
python -m polyfempy.api.examples.parameter_sweep
```

**Use case:** Useful for sensitivity analysis, parameter studies, and optimization.

---

### 5. `batch_processing.py`
**Batch processing with error isolation**

Demonstrates how to:
- Run multiple simulations efficiently using `batch_solve()`
- Handle errors gracefully (one failure doesn't stop others)
- Process results from multiple jobs

**Run:**
```bash
python -m polyfempy.api.examples.batch_processing
```

**Use case:** Parameter sweeps, Monte Carlo simulations, or processing multiple configurations.

---

### 6. `with_callbacks.py`
**Monitoring simulation progress**

Shows how to:
- Monitor simulation results and metadata
- Study convergence by running simulations with different parameters
- Analyze convergence behavior
- Generate convergence plots

**Run:**
```bash
python -m polyfempy.api.examples.with_callbacks
```

**Note:** Callbacks are not directly supported in the high-level `solve()` API, but you can monitor progress by checking result metadata and running parameter studies.

**Use case:** Understanding convergence behavior, debugging, or analyzing simulation results.

---

## Running All Examples

To run all examples:

```bash
# Windows PowerShell
Get-ChildItem polyfempy\api\examples\*.py | Where-Object { $_.Name -ne '__init__.py' } | ForEach-Object { python -m polyfempy.api.examples.$($_.BaseName) }

# Linux/Mac
for file in polyfempy/api/examples/*.py; do
    if [ "$(basename $file)" != "__init__.py" ]; then
        python -m polyfempy.api.examples.$(basename $file .py)
    fi
done
```

---

## Requirements

- Python 3.7+
- NumPy
- PolyFEM Python API (`polyfempy.api`)

Optional:
- Matplotlib (for convergence plots in `with_callbacks.py`)
- Meshio (for VTU export in `run_elasticity.py`)

---

## Next Steps

After running these examples, you can:

1. **Modify parameters** to experiment with different configurations
2. **Use your own meshes** by replacing the simple unit square
3. **Combine techniques** from different examples
4. **Check the API documentation** in `docs/api-architecture.md` for more details

For differentiable simulations (shape optimization, material optimization), see:
- `polyfempy/differentiable/examples/` directory
- `docs/differentiable-guide.md`

