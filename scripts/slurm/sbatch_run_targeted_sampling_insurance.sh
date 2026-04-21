#!/bin/bash
# =============================================================================
# Slurm: targeted sampling + insurance (see targeted_sampling_insurance_cases.py).
#
# Default: 23 array tasks (shards), ~100 cases each, up to 4 parallel PolyFEM **processes**
# per node. Must match ``DEFAULT_NUM_SHARDS`` in that module if you change RAW_CASES.
#
# Resource hint: 4 concurrent solves × OMP=1 → ``--cpus-per-task=4``; raise ``--mem``
# if OOM (try 48G–64G if needed).
#
# Per-process single-thread (recommended on shared CC nodes)::
#   export OMP_NUM_THREADS=1
#   export MKL_NUM_THREADS=1
#
# Submit from **polyfem-python repo root**::
#   mkdir -p slurm_logs
#   sbatch scripts/slurm/sbatch_run_targeted_sampling_insurance.sh
#
# Force re-run (ignore success rows in each shard CSV)::
#   sbatch --export=ALL,EXTRA_PY_ARGS='--force' scripts/slurm/sbatch_run_targeted_sampling_insurance.sh
#
# If you change shard count, edit NUM_SHARDS and #SBATCH --array=0-(N-1) together.
# =============================================================================

#SBATCH --job-name=ts_ins

#SBATCH --output=slurm_logs/slurm-%A_%a-ts_ins.out
#SBATCH --error=slurm_logs/slurm-%A_%a-ts_ins.err
#SBATCH --array=0-22

#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=def-teseo

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/scratch/polyfem-python}"
cd "$REPO" || {
  echo "ERROR: cannot cd to REPO=$REPO"
  exit 1
}

mkdir -p slurm_logs

PY_SCRIPT="experiment/new_experiment/theta_degree/run_targeted_sampling_insurance_sharded.py"
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
echo "[sbatch] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"

NUM_SHARDS="${NUM_SHARDS:-23}"

echo "[sbatch] array task shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"

EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"

# shellcheck disable=SC2086
exec "$PY" "$PY_SCRIPT" --shards "$NUM_SHARDS" --shard "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" \
  --max-concurrent 4 ${EXTRA_PY_ARGS}
