#!/bin/bash
# =============================================================================
# Slurm: piecewise-theta full sweep (theta_piecewise_all_cases / sharded runner).
#
# Default: 135 array tasks (shards), each running ~94 cases with up to 5 parallel
# PolyFEM **processes** per node. Must match ``DEFAULT_NUM_SHARDS`` in
# ``theta_piecewise_all_cases.py`` (if you change ``RAW_CASES``, refresh both).
#
# Resource hint: 5 concurrent solves × OMP=1 → ``--cpus-per-task=5`` is a sensible
# starting point; raise ``--mem`` if OOM (try 48G–64G if solves are heavy).
#
# Per-process single-thread (recommended on shared CC nodes)::
#   export OMP_NUM_THREADS=1
#   export MKL_NUM_THREADS=1
#
# Submit from **polyfem-python repo root**::
#   mkdir -p slurm_logs
#   sbatch scripts/slurm/sbatch_run_theta_piecewise.sh
#
# Force re-run (ignore success rows in each shard CSV)::
#   sbatch --export=ALL,EXTRA_PY_ARGS='--force' scripts/slurm/sbatch_run_theta_piecewise.sh
#
# If you change shard count::
#   export NUM_SHARDS=200
#   sbatch --array=0-199 scripts/slurm/sbatch_run_theta_piecewise.sh
#   (edit #SBATCH --array inside this file to match 0-(NUM_SHARDS-1).)
# =============================================================================

#SBATCH --job-name=theta_pcw

#SBATCH --output=slurm_logs/slurm-%A_%a-theta_pcw.out
#SBATCH --error=slurm_logs/slurm-%A_%a-theta_pcw.err
#SBATCH --array=0-134

#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --account=def-teseo

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/scratch/polyfem-python}"
cd "$REPO" || {
  echo "ERROR: cannot cd to REPO=$REPO"
  exit 1
}

mkdir -p slurm_logs

PY_SCRIPT="experiment/new_experiment/theta_degree/run_theta_piecewise_sharded.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: missing $REPO/$PY_SCRIPT"
  exit 1
fi

echo "[sbatch] host=$(hostname) repo=$REPO jobid=${SLURM_JOB_ID:-local}"

module load StdEnv/2023

if [[ -f "${HOME}/polyfem_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/polyfem_env/bin/activate"
else
  echo "ERROR: missing ${HOME}/polyfem_env/bin/activate"
  exit 1
fi

hash -r

if [[ -f "${HOME}/env_polyfem_gmsh.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/env_polyfem_gmsh.sh"
fi

PY="${HOME}/polyfem_env/bin/python"
"$PY" -c "import sys; print('python:', sys.executable)"

# One OS thread per worker process (×5 workers ≈ 5 CPUs).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
echo "[sbatch] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"

NUM_SHARDS="${NUM_SHARDS:-135}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[sbatch] array task shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
fi

EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"

# shellcheck disable=SC2086
exec "$PY" "$PY_SCRIPT" --shards "$NUM_SHARDS" --shard "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" \
  --max-concurrent 5 ${EXTRA_PY_ARGS}
