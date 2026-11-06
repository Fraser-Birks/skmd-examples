#!/bin/bash
#SBATCH --job-name=ad_test_set
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:00:00

# EXAMPLE SLURM script (Quadro RTX 6000 GPU, 4 cores): adapt the #SBATCH lines to your site and set
# SKMD_MODULES to the modules your builds need (README, "Cluster environment").

# Build the test set (paper App. E.2: oracle-constrained relaxations on a 10x10 (phi, psi) grid).
# Submit from the repository root; for large grids use an array + N_SHARDS, then concatenate shards.
#
# Usage:
#   S=systems/alanine_dipeptide/slurm/example/generate_test_set.slurm
#   sbatch $S                                        # paper setting (10x10)
#   sbatch --export=ALL,NPHI=20,NPSI=20 $S
#   sbatch --export=ALL,N_SHARDS=4 --array=0-3 $S
#   sbatch --export=ALL,N_RATTLED=500 $S   # also add off-equilibrium rattled points (not used in the paper)

# Optional: SKMD_MODULES = space-separated module list to load.
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

VENV_PATH="${VENV_PATH:-${SKMD_VENV:?set SKMD_VENV (or VENV_PATH) to your venv}}"
source "${VENV_PATH}/bin/activate"

set -euo pipefail

# $0 is SLURM's spool copy, so default to the submit dir (repo root); override via --export.
SCRIPT_ROOT="${SCRIPT_ROOT:-${SLURM_SUBMIT_DIR}/systems/alanine_dipeptide}"
FILE_ROOT="${FILE_ROOT:-${SLURM_SUBMIT_DIR}/data/alanine_dipeptide}"

NPHI="${NPHI:-10}"
NPSI="${NPSI:-10}"
N_SHARDS="${N_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
N_RATTLED="${N_RATTLED:-0}"
RATTLE_STDEV_MIN="${RATTLE_STDEV_MIN:-0.01}"
RATTLE_STDEV_MAX="${RATTLE_STDEV_MAX:-0.15}"
RATTLE_SEED="${RATTLE_SEED:-0}"

INIT_XYZ="${FILE_ROOT}/ace-ala-nme.xyz"
ORACLE_MODEL="${FILE_ROOT}/MACE-OFF23_small.model"
OUTPUT="${FILE_ROOT}/test_set.xyz"

echo "Grid: ${NPHI} x ${NPSI}   shard ${SHARD_INDEX}/${N_SHARDS}"
echo "Output: ${OUTPUT}"

python "${SCRIPT_ROOT}/setup/generate_test_set.py" \
    --init_xyz "${INIT_XYZ}" \
    --oracle_model "${ORACLE_MODEL}" \
    --output "${OUTPUT}" \
    --device cuda \
    --nphi "${NPHI}" \
    --npsi "${NPSI}" \
    --n_shards "${N_SHARDS}" \
    --shard_index "${SHARD_INDEX}" \
    --n_rattled "${N_RATTLED}" \
    --rattle_stdev_min "${RATTLE_STDEV_MIN}" \
    --rattle_stdev_max "${RATTLE_STDEV_MAX}" \
    --rattle_seed "${RATTLE_SEED}"
