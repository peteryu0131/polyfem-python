#!/bin/bash
# =============================================================================
# Slurm: theta-degree Batch 2 (theta_batch2_cases / sharded runner).
#
# Default: 33 array tasks (shards), each running 3240/33≈98 cases with up to 5
# parallel PolyFEM **processes** per node.
#
# Submit from **polyfem-python repo root**::
#   mkdir -p slurm_logs
#   sbatch scripts/slurm/sbatch_run_theta_batch2.sh
#
# Force re-run::
#   sbatch --export=ALL,EXTRA_PY_ARGS='--force' scripts/slurm/sbatch_run_theta_batch2.sh
# =============================================================================

#SBATCH --job-name=theta_b2

#SBATCH --output=slurm_logs/slurm-%A_%a-theta_b2.out
#SBATCH --error=slurm_logs/slurm-%A_%a-theta_b2.err
#SBATCH --array=0-32

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

PY_SCRIPT="experiment/new_experiment/theta_degree/run_theta_batch2_sharded.py"
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

NUM_SHARDS="${NUM_SHARDS:-33}"
echo "[sbatch] array task shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"

EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"

# shellcheck disable=SC2086
exec "$PY" "$PY_SCRIPT" --shards "$NUM_SHARDS" --shard "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" \
  --max-concurrent 5 ${EXTRA_PY_ARGS}

