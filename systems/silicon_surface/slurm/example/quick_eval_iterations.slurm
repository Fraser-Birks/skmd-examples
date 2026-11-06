#!/bin/bash
#SBATCH --job-name=si_quick_eval
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:15:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# CPU-only reconstruction-energy check across AL iterations.
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"
: "${RUN_DIR:?RUN_DIR not set}"
: "${REFERENCE_XYZ:?REFERENCE_XYZ not set}"
: "${RECONSTRUCTED_XYZ:?RECONSTRUCTED_XYZ not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
REFERENCE_DFT_DIR="${REFERENCE_DFT_DIR:-}"
RECONSTRUCTED_DFT_DIR="${RECONSTRUCTED_DFT_DIR:-}"

ARGS=(
    --run-dir "${RUN_DIR}"
    --reference-xyz "${REFERENCE_XYZ}"
    --reconstructed-xyz "${RECONSTRUCTED_XYZ}"
    --output-dir "${OUTPUT_DIR}"
    --device cpu
)
if [[ -n "${REFERENCE_DFT_DIR}" ]]; then
    ARGS+=(--reference-dft-dir "${REFERENCE_DFT_DIR}" --reconstructed-dft-dir "${RECONSTRUCTED_DFT_DIR}")
fi

echo "=== quick_eval_iterations: reconstruction-energy check across iterations ==="
python "${SCRIPT_ROOT}/analysis/quick_eval_iterations.py" "${ARGS[@]}"
echo "quick_eval_iterations complete."
