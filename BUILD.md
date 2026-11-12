# Build the Python bindings

From the repository root, initialize the pinned PolyFEM sources, data, examples and generator:

```sh
git submodule update --init --recursive
```

Use an environment with Python 3.10 or later, CMake 3.25 or later, and a C++ compiler. On Windows, use an x64 Visual Studio development environment. To use an existing conda environment named `polyfem`, activate it with `conda activate polyfem` first.

## Development install

```sh
python -m pip install "setuptools>=77" wheel "cmake>=3.25" "nanobind>=2" numpy pytest
python tools/generate_polyfem_api.py
python -m pip install -e . --no-build-isolation
python -m polyfempy backend-info --require
```

Regenerate the API after updating the schemas, generator or generator configuration. Generation writes to `polyfempy/generated_api/` and may download the linked solver schemas. The first native build also needs access to the pinned third-party dependencies.

## Build and check a wheel

From a fresh checkout with the development dependencies and generated API in place, run:

```sh
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

In a separate environment with a compatible Python version and architecture, install the resulting wheel by its full path. Change to a directory outside the repository and run the example in [README.md](README.md). For the native reference check and full example sweep, see [TESTING.md](TESTING.md).
