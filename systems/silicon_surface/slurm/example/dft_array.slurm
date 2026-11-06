#!/bin/bash
#SBATCH --job-name=si_al_dft
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:30:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# CPU VASP fallback (--cpu-dft): 12 pure-MPI ranks; KPAR scales best here (config_cpu.yaml: KPAR=6, NCORE=2).

if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

export OMP_NUM_THREADS=1
# Cheap insurance against default-stack-size crashes.
ulimit -s unlimited

# CPU (MPI) build of vasp_std. Falls back to VASP_STD if VASP_STD_CPU is unset.
VASP="${VASP_STD_CPU:-${VASP_STD:?set VASP_STD_CPU (or VASP_STD) to the path of your CPU vasp_std binary}}"

: "${DFT_INPUT_DIR:?DFT_INPUT_DIR must be set}"
FRAME=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
FRAME_DIR="${DFT_INPUT_DIR}/frame_${FRAME}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${FRAME_DIR}"
cd "${FRAME_DIR}"

# Log a nonzero exit but continue: dft_runner.py checks OUTCAR itself.
set +e
srun "${VASP}" > vasp.out 2>&1
VASP_EXIT=$?
if [[ "${VASP_EXIT}" -ne 0 ]]; then
    echo "WARNING: vasp_std exited ${VASP_EXIT} -- check vasp.out/OUTCAR before assuming this frame failed (see comment above)"
fi
echo "Finished: $(date)"
