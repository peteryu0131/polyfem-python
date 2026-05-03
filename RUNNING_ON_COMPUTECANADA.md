# Running On Compute Canada / Alliance

This note describes the expected environment for running this repository on
Compute Canada / Digital Research Alliance of Canada clusters. It is meant for
fresh login shells, interactive jobs, and Slurm batch jobs.

The examples below assume:

- repository: `$HOME/scratch/polyfem-python`
- virtual environment: `$HOME/polyfem_env`
- optional Gmsh SDK environment script: `$HOME/env_polyfem_gmsh.sh`

Adjust those paths if your checkout or virtual environment lives elsewhere.

## Rules To Follow

1. Load modules before activating the virtual environment.
2. Run `hash -r` after changing `PATH` in bash.
3. Use the virtual-environment Python explicitly in scripts when possible:
   `"$HOME/polyfem_env/bin/python"`.
4. Source the Gmsh SDK environment before running workflows that import
   `gmsh` or generate meshes.
5. Keep generated `runs/`, `slurm_logs/`, and output files out of git.

Check the active Python whenever an import looks wrong:

```bash
which python
python -c "import sys; print(sys.executable)"
```

Both commands should point at `$HOME/polyfem_env/bin/python` after activation.

## One-Time Setup

Create the virtual environment:

```bash
module load StdEnv/2023
python3 -m venv "$HOME/polyfem_env"
```

Install the basic build/runtime dependencies:

```bash
unset PIP_CONFIG_FILE
"$HOME/polyfem_env/bin/python" -m pip install -U pip setuptools wheel
"$HOME/polyfem_env/bin/python" -m pip install nanobind numpy Cython
```

Install optional dependencies as needed:

```bash
# VTU/mesh IO helpers
"$HOME/polyfem_env/bin/python" -m pip install meshio

# Differentiable examples and optimization helpers
"$HOME/polyfem_env/bin/python" -m pip install torch
```

If `torch` is not available from the active package indexes, try the CPU wheel
index:

```bash
"$HOME/polyfem_env/bin/python" -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
```

For `shapely` builds on Alliance systems, load GEOS first:

```bash
module load StdEnv/2023 geos
"$HOME/polyfem_env/bin/python" -m pip install --no-build-isolation "shapely>=2"
```

Build and install this repository from the repository root:

```bash
module load StdEnv/2023 cmake
cd "$HOME/scratch/polyfem-python"
unset PIP_CONFIG_FILE
"$HOME/polyfem_env/bin/python" -m pip install -e . --no-build-isolation
```

## Gmsh SDK

Some paper and mesh-generation workflows import `gmsh`. On Alliance systems,
using the official Linux SDK is usually more reliable than relying on the PyPI
`gmsh` wheel.

Create `$HOME/env_polyfem_gmsh.sh` after unpacking the SDK:

```bash
cat > "$HOME/env_polyfem_gmsh.sh" <<'EOF'
export GMSH_SDK="$HOME/opt/gmsh/gmsh-4.x.x-Linux64-sdk"
export PYTHONPATH="$GMSH_SDK/lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$GMSH_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$GMSH_SDK/bin${PATH:+:$PATH}"
EOF
chmod +x "$HOME/env_polyfem_gmsh.sh"
```

Replace `gmsh-4.x.x-Linux64-sdk` with the directory you actually unpacked.

You may also append this block to `$HOME/polyfem_env/bin/activate`:

```bash
if [ -f "$HOME/env_polyfem_gmsh.sh" ]; then
    . "$HOME/env_polyfem_gmsh.sh"
fi
```

Even if you do that, it is fine to source the Gmsh script explicitly in Slurm
jobs. That makes batch jobs less dependent on interactive-shell details.

## New Shell Checklist

Run this at the start of a login shell or an interactive allocation:

```bash
module load StdEnv/2023 geos
source "$HOME/polyfem_env/bin/activate"
hash -r

# Required for workflows that import gmsh or generate meshes.
source "$HOME/env_polyfem_gmsh.sh"

cd "$HOME/scratch/polyfem-python"

which python
python -c "import sys; print(sys.executable)"
python -c "import numpy, polyfempy; print('base imports OK')"
```

For mesh-generation workflows, also check:

```bash
python -c "import gmsh, shapely; print('mesh imports OK')"
```

## Public Examples

The top-level examples do not require Compute Canada, but they are a good
environment smoke test after building `polyfempy`:

```bash
python examples/01_forward_solve.py
python examples/02_result_fields.py
```

Differentiable examples additionally require PyTorch:

```bash
python examples/03_shape_gradient.py
python examples/05_parameterized_vertex_map.py
```

Outputs are written under `examples/runs/`, which is ignored by git.

## Paper Smoke Runs

The current paper-facing scripts live under `experiment/paper_experiment/`.

Run one short local E/h/theta smoke case:

```bash
PAPER_HTHETA_FRAMES=0 PAPER_HTHETA_OPT_STEPS=1 \
PAPER_HTHETA_TEND=0.02 PAPER_HTHETA_DT=0.01 \
python experiment/paper_experiment/compute_canada_run_test/run_cases.py \
  --cases experiment/paper_experiment/compute_canada_run_test/cases_smoke.json \
  --case-index 0 \
  --run-name local_smoke
```

Run one short 07 h/theta case:

```bash
PAPER_HTHETA_OPT_STEPS=3 PAPER_HTHETA_TEND=0.02 PAPER_HTHETA_DT=0.01 \
python experiment/paper_experiment/compute_canada_run_07/run_cases.py \
  --case-index 0 \
  --run-name local_07_smoke
```

## Slurm Submission

Create log directories before submitting:

```bash
mkdir -p experiment/paper_experiment/compute_canada_run_test/slurm_logs
mkdir -p experiment/paper_experiment/compute_canada_run_07/slurm_logs
```

Submit the current paper smoke arrays:

```bash
sbatch experiment/paper_experiment/compute_canada_run_test/sbatch_E_h_theta_tests.sh
sbatch experiment/paper_experiment/compute_canada_run_07/sbatch_07_h_theta_cases.sh
```

The scripts default to:

- `REPO_ROOT="${SLURM_SUBMIT_DIR:-$HOME/scratch/polyfem-python}"`
- `VENV_PATH="$HOME/polyfem_env"`
- `GMSH_ENV="$HOME/env_polyfem_gmsh.sh"`

Override those variables at submission time if needed:

```bash
sbatch --export=ALL,REPO_ROOT="$HOME/scratch/polyfem-python",VENV_PATH="$HOME/polyfem_env" \
  experiment/paper_experiment/compute_canada_run_test/sbatch_E_h_theta_tests.sh
```

## Minimal Slurm Template

Use this shape for new batch scripts:

```bash
#!/bin/bash
#SBATCH --job-name=polyfem_example
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

module load StdEnv/2023 geos

REPO_ROOT="${REPO_ROOT:-$HOME/scratch/polyfem-python}"
VENV_PATH="${VENV_PATH:-$HOME/polyfem_env}"
GMSH_ENV="${GMSH_ENV:-$HOME/env_polyfem_gmsh.sh}"

source "$VENV_PATH/bin/activate"
hash -r

if [ -f "$GMSH_ENV" ]; then
    source "$GMSH_ENV"
fi

cd "$REPO_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"$VENV_PATH/bin/python" examples/01_forward_solve.py
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: polyfempy` | Check `which python`; reinstall with `"$HOME/polyfem_env/bin/python" -m pip install -e . --no-build-isolation`. |
| `which python` points to `/cvmfs/...` | Load modules first, then `source "$HOME/polyfem_env/bin/activate"`, then `hash -r`. |
| `No module named gmsh` | Source `$HOME/env_polyfem_gmsh.sh` or fix `GMSH_SDK` in that file. |
| `No module named shapely` or missing `geos_c.h` | `module load geos`, then reinstall shapely in the venv. |
| `torch` cannot be installed | Try the Alliance wheelhouse first; if needed use the PyTorch CPU wheel index shown above. |
| VTU reading fails with missing `meshio` | Install `meshio` in the venv or install `polyfempy[io]`. |
| Batch job works locally but not under Slurm | Use explicit `VENV_PATH`, `GMSH_ENV`, `REPO_ROOT`, and call `"$VENV_PATH/bin/python"` in the script. |

## Path Summary

| Item | Typical path |
| --- | --- |
| Repository | `$HOME/scratch/polyfem-python` or `/scratch/$USER/polyfem-python` |
| Virtual environment | `$HOME/polyfem_env` |
| Gmsh environment script | `$HOME/env_polyfem_gmsh.sh` |
| Public example outputs | `examples/runs/` |
| Paper run outputs | `experiment/paper_experiment/**/runs/` |
| Slurm logs | `experiment/paper_experiment/**/slurm_logs/` |
