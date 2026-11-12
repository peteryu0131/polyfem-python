# Testing the forward API

Run these commands in the repository root after initializing the submodules. See [BUILD.md](BUILD.md) for the native build.

## Python and example tests

Generate the Python API before running the tests. For a checkout without an installed package, add the public package directory to the search path. In PowerShell:

```powershell
python -m pip install numpy pytest
python tools/generate_polyfem_api.py
$env:PYTHONPATH = (Resolve-Path python).Path
python -m pytest tests -q
```

On Linux or macOS, use `PYTHONPATH="$PWD/python" python -m pytest tests -q`. The suite includes comparisons between classic example configurations and official JSON inputs.

## Native solver

With the compiled package installed, check that the backend loads, then solve the Neo-Hookean cube and compare its six error metrics with the official reference.

```sh
python -m polyfempy backend-info --require
python -m pytest tests/test_backend_smoke.py -q
```

## GitHub Actions

| When | What runs |
| --- | --- |
| Push or pull request | Python and generated API tests on Windows, macOS and Linux with Python 3.10–3.12 |
| Draft PR, or PR targeting another branch | Linux C++ build and the cube reference check on open, update or reopen |
| Ready-for-review PR targeting `main` | Cube check followed by the full contact sweep on open, update, reopen, marking ready or retargeting to `main` |
| Manual run of Generated Contact Backend, or push to main | Full contact sweep against the pinned official data |

Before merging, check **Forward solver checks (Ubuntu, Python 3.11)** on the latest PR revision. The full sweep runs the 75 active cases and compares the generated configurations with the official inputs and their reference metrics. For an on-demand check, open **Actions → Generated Contact Backend → Run workflow** and select the branch. To run the same check locally:

```sh
python tools/run_generated_contact_backend_checks.py --output-root build/contact-check --generated-source-tolerance 1e-5 --max-threads 1 --require-tests-match
```

Read `build/contact-check/summary.txt` for the outcome and the per-case logs for failures. In GitHub Actions, download the **generated-contact-backend-results** artifact from the workflow run to inspect those files.
