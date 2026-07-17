# Build And Installation Notes

This file contains development/build notes for PolyFEM-Python. The root
`README.md` is intentionally focused on the public API and artifact story.

The Python bindings are still evolving. Expect API changes and possible bugs.

This repository builds a C++ extension module named `polyfempy.polyfempy`
compiled with nanobind. In Python, the package is imported as:

```python
import polyfempy as pf
```

## Repository Layout Direction

The current checkout contains the Python binding, generator, data, and examples
in one working tree. The future direction is to keep these as separate repos
that are checked out at stable paths under `polyfem-python`, likely through Git
submodules:

```text
polyfem-python/
  polyfempy/
  src/
  generator-config/
  python-from-jse/
  polyfem-data/
  examples/
```

`generator-config/` is reserved for PolyFEM-specific generator config. The
active config lives there now, while `python-from-jse/` stays focused on the
generic generator and its tools.

Generated Python API files are written to `polyfempy/generated/` during explicit
regeneration. They should be treated as generated artifacts, not as
hand-maintained source files.

## Conda-Forge Package Versus Source Build

There are two distinct ways to consume `polyfempy`.

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
- The high-level Python entry point is `polyfempy.api.solve`.
- The C++ binding implements `Solver.solve()` as returning `(sol, pressure)`.
- Use `polyfempy.api.Result` for structured Python-side result fields.
