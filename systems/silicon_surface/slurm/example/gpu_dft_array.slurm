#!/bin/bash
#SBATCH --job-name=si_al_dft_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=04:00:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# GPU VASP, one rank/GPU per task (~1.2 GPU-h per SCAN single point); config: dft/config_gpu.yaml.

if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

# Extra runtime library dirs for your VASP build, if any (e.g. NVHPC qd/lib).
if [[ -n "${VASP_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${VASP_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

# GPU (OpenACC) build of vasp_std (the paper used VASP 6.5.1).
VASP="${VASP_STD:?set VASP_STD to the path of your GPU vasp_std binary}"

: "${DFT_INPUT_DIR:?DFT_INPUT_DIR must be set}"
FRAME=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
FRAME_DIR="${DFT_INPUT_DIR}/frame_${FRAME}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${FRAME_DIR}"
cd "${FRAME_DIR}"

# This build can segfault in MPI teardown after writing a complete OUTCAR; log the exit
# code and continue (dft_runner.py _has_final_energy_block checks completion).
set +e
"${VASP}" > vasp.out 2>&1
VASP_EXIT=$?
if [[ "${VASP_EXIT}" -ne 0 ]]; then
    echo "WARNING: vasp_std exited ${VASP_EXIT} -- check vasp.out/OUTCAR before assuming this frame failed (see comment above)"
fi
echo "Finished: $(date)"
