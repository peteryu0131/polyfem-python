# polyfempy (PolyFEM Python bindings)

The Python bindings are **alpha**. Expect API changes and possible bugs.

This repository builds a C++ extension module named **`polyfempy`** (compiled with **nanobind**). In Python, you should always:

```python
import polyfempy as pf
```

---

## Why your instructor says “conda install conda-forge::polyfempy”

There are two distinct ways to consume `polyfempy`:

### 1) Conda-forge package (deployment/tutorial path)

- `conda install conda-forge::polyfempy`
- Pros: prebuilt binaries (no local C++ toolchain hassles), reproducible environment.
- Cons: **deployment is slower** because conda-forge builds for multiple platforms, runs CI, applies patches, and then publishes. The packaged version may lag behind the latest repository commits.

That’s why a tutorial often follows the **conda-forge deployed version**: everyone can install the same version reliably.

### 2) Build from source (latest features / fast iteration)

If you “want the juicy latest feature”, you build from source and install via pip. This compiles the C++ extension on your machine using your local toolchain (CMake + MSVC on Windows).

Historically instructors might say:

- `python setup.py install`
- `python setup.py test`

That works for some setups, but the modern/recommended workflow is via `pip`, which still uses `setup.py`/CMake under the hood.

---

## Build & install from source (recommended commands)

### Windows (Visual Studio Developer PowerShell)

From the repository root:

```powershell
python -m pip install -U pip setuptools wheel cmake nanobind
python -m pip install -e . --no-build-isolation
```

To log the full build output:

```powershell
python -m pip install -e . --no-build-isolation *>&1 | Tee-Object -FilePath .\build.log
```

Verify installation:

```powershell
python -c "import polyfempy as pf; print(pf.version())"
```

---

## Notes about the runtime API

- The core compute happens in C++ (`pf.Solver()`).
- The C++ binding implements `Solver.solve()` as returning a tuple: **`(sol, pressure)`**.

