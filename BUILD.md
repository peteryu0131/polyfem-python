# Build And Installation Notes

This file contains development/build notes for PolyFEM-Python. The root
`README.md` is intentionally focused on the public API and artifact story.

The Python bindings are still evolving. Expect API changes and possible bugs.

This repository builds a C++ extension module named `polyfempy.polyfempy`
compiled with nanobind. In Python, the package is imported as:

```python
import polyfempy as pf
```

For the full fresh-clone, install, CI, backend smoke, and generated-example
diagnostic workflow, see `TESTING.md`.

## Repository Layout Direction

The current checkout contains the Python binding, generator, data, and examples
in one working tree. The future direction is to keep these as separate repos
that are checked out at stable paths under `polyfem-python`, likely through Git
submodules:

```text
polyfem-python/
  polyfempy/
  src/
  polyfem/
  generator-config/
  python-from-jse/
  polyfem-data/
  examples/
```

`polyfem/` is the PolyFEM source submodule. It provides the backend source and
canonical JSON specs. `generator-config/` is reserved for PolyFEM-specific
Python API config. The active config lives there now, while `python-from-jse/`
stays focused on the generic generator, its tools, and dummy examples.

For a fresh checkout, clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/polyfem/polyfem-python.git
cd polyfem-python
```

If the repository was already cloned without submodules, initialize them before
generating or building:

```powershell
git submodule update --init --recursive
```

The CMake build uses the `polyfem/` submodule directly. It does not fetch
PolyFEM through CPM. If the submodule is missing, configure fails with the
submodule command above.

Generated Python API files are written to `polyfempy/generated_api/` during
explicit regeneration. They should be treated as generated artifacts, not as
hand-maintained source files.

Regenerate them from the repository root:

```powershell
python tools\generate_polyfem_api.py
```

Run backend-free generated API checks from the repository root:

```powershell
python tools\generate_polyfem_api.py --check
```

These checks regenerate the API, compile/import the generated Python layer
through the generator test suite, and run classic example source-JSON parity
tests. They do not run backend simulations.

## Conda-Forge Package Versus Source Build

There are two distinct ways to consume `polyfempy`.

## Local Conda Build Checklist

Use this when you want to test the current checkout against the real C++
backend. This is the path for your local `polyfem` conda environment.

```powershell
conda activate polyfem
git submodule update --init --recursive
python -m pip install -U pip setuptools wheel cmake nanobind pytest numpy
python tools\generate_polyfem_api.py
$env:N_THREADS = "4"
python -m pip install -e . --no-build-isolation *>&1 | Tee-Object -FilePath .\build.log
python -m polyfempy backend-info --require
```

After the backend is available, run the backend smoke test:

```powershell
python -m pytest tests\test_backend_smoke.py -q
```

For normal Python/generator checks that do not build or load the C++ backend:

```powershell
python tools\generate_polyfem_api.py
python -m pytest tests -q
```

The command-line entry point is intentionally small:

```powershell
python -m polyfempy backend-info
python -m polyfempy solve path\to\input.json --output-dir out
```

`backend-info` is safe before building. `solve` requires the compiled
`polyfempy.polyfempy` extension.

### Conda-Forge Package

```bash
conda install conda-forge::polyfempy
```

Pros:

- prebuilt binaries
- fewer local C++ toolchain issues
- reproducible environment for tutorials

Cons:

- the packaged version may lag behind the latest repository commits
- conda-forge deployment is slower because builds run across platforms

### Source Build

Use this path for the latest repository features and active development. It
compiles the C++ extension locally with CMake and nanobind.

Linux/macOS:

```bash
python -m pip install -U pip setuptools wheel cmake nanobind
python -m pip install -e . --no-build-isolation
```

Windows, from Visual Studio Developer PowerShell:

```powershell
python -m pip install -U pip setuptools wheel cmake nanobind
python -m pip install -e . --no-build-isolation
```

To log the full Windows build output:

```powershell
python -m pip install -e . --no-build-isolation *>&1 | Tee-Object -FilePath .\build.log
```

Verify installation:

```bash
python -c "import polyfempy as pf; print(pf.cpp_backend_available()); print(pf.cpp_backend_error())"
```

Expected backend status:

```text
True
None
```

If the backend is unavailable, `import polyfempy` can still succeed and
pure-Python config/import tests can still run, but real solves require the C++
backend.

## Runtime API Notes

- The core compute happens in C++ through `pf.Solver()`.
- The high-level Python entry point is `polyfempy.runtime.solve`.
- The C++ binding implements `Solver.solve()` through the new VarForm
  `polyfem::State::solve(sol)` path.
- `polyfempy.runtime.Result.sol` is the raw backend solution. It is not assumed
  to be aligned with mesh vertices or sampled visualization data.
- File outputs are owned by the backend `output` config and VarForm
  `save_json(sol)` / `export_data(sol)` calls.
