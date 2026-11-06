#!/bin/bash
#SBATCH --job-name=si_al_uhal_p1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=20:00:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# Pure ASE + mace-torch on one GPU; committee bias runs ~7.8 steps/s (~3.5 h MD), hence 7 h.

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"          # .../silicon_surface/uhal_<label>
: "${FILE_ROOT:?FILE_ROOT not set}"      # .../silicon_surface
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"  # .../systems/silicon_surface
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-30}"

RUN_NAME="${RUN_NAME:-uhal}"
NSTEPS="${NSTEPS:-200}"
N_RUN="${N_RUN:-500}"
COMMITTEE_SIZE="${COMMITTEE_SIZE:-10}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.5}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-10.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
TAU_REL="${TAU_REL:-0.03}"
TAU_HIST="${TAU_HIST:-10}"
SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD:-adaptive}"
SELECTOR_DELAY="${SELECTOR_DELAY:-10}"
SELECTOR_MIXING="${SELECTOR_MIXING:-0.1}"
SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL:-1.2}"
SELECTOR_INTERVAL="${SELECTOR_INTERVAL:-2}"
T_VAL="${T_VAL:-700.0}"
FRICTION="${FRICTION:-0.01}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
# Burn-in yields N = N_BURNIN_MD // EVERY_BURNIN walkers, matching the SKMD ensemble.
N_SILENT_BURNIN="${N_SILENT_BURNIN:-4000}"
N_BURNIN_MD="${N_BURNIN_MD:-6000}"
EVERY_BURNIN="${EVERY_BURNIN:-200}"
N_FROZEN="${N_FROZEN:?N_FROZEN not set — see prepare_lammps_data.py}"
GPU_DFT="${GPU_DFT:-1}"
SKIP_MD="${SKIP_MD:-}"

LAMMPS_DATA="${FILE_ROOT}/silicon_slab.data"
INIT_XYZ="${FILE_ROOT}/relaxed_struct_non_pandey.xyz"

SIM_DIR="${AL_ROOT}/simulation/iteration_${ITER}"
mkdir -p "${SIM_DIR}"
cd "${SIM_DIR}"

echo "=== UHAL Phase 1: iteration ${ITER} / ${N_ITERS} ==="
echo "Model: ${MODEL_NAME}   Selections: ${N_SELECT}"

# Locate the foundation model for this iteration.
if [[ "${ITER}" -eq 1 ]]; then
    MODEL_SRC="${FILE_ROOT}/${MODEL_NAME}.model"
else
    MODEL_SRC="${AL_ROOT}/training/iteration_$((ITER - 1))/${MODEL_NAME}.model"
fi
cp "${MODEL_SRC}" .

# No LAMMPS/symmetrix needed: uses the MACE descriptor directly.

# run_uhal_mace.py writes diverse_subset.xyz last, in both modes.
COMPLETION_MARKER="${SIM_DIR}/${RUN_NAME}/diverse_subset.xyz"
if [[ -n "${SKIP_MD}" ]]; then
    # SKIP_MD=1 (via --export): MD already completed, jump to writing DFT inputs.
    [[ -f "${COMPLETION_MARKER}" ]] || {
        echo "SKIP_MD set but ${COMPLETION_MARKER} not found — nothing to skip to, aborting" >&2
        exit 1
    }
    echo "SKIP_MD set and ${COMPLETION_MARKER} exists — skipping run_uhal_mace.py"
    # --export=ALL downstream: unset so SKIP_MD does not leak into later jobs.
    unset SKIP_MD
else
    python "${SCRIPT_ROOT}/uhal_run/run_uhal_mace.py" \
        --lammps_data "${LAMMPS_DATA}" \
        --init_xyz "${INIT_XYZ}" \
        --mace_model "${MODEL_NAME}.model" \
        --n_frozen "${N_FROZEN}" \
        --fname "${RUN_NAME}" \
        --nsteps "${NSTEPS}" \
        --n_run "${N_RUN}" \
        --n_select "${N_SELECT}" \
        --committee_size "${COMMITTEE_SIZE}" \
        --prior_weight "${PRIOR_WEIGHT}" \
        --energy_weight "${ENERGY_WEIGHT}" \
        --forces_weight "${FORCES_WEIGHT}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --tau_rel "${TAU_REL}" \
        --tau_hist "${TAU_HIST}" \
        --selector_threshold "${SELECTOR_THRESHOLD}" \
        --selector_delay "${SELECTOR_DELAY}" \
        --selector_mixing "${SELECTOR_MIXING}" \
        --selector_thresh_mul "${SELECTOR_THRESH_MUL}" \
        --selector_interval "${SELECTOR_INTERVAL}" \
        --T "${T_VAL}" \
        --friction "${FRICTION}" \
        --timestep_fs "${TIMESTEP_FS}" \
        --n_silent_burnin "${N_SILENT_BURNIN}" \
        --n_burnin_md "${N_BURNIN_MD}" \
        --every_burnin "${EVERY_BURNIN}"
fi

# ── Write VASP inputs for the newly selected structures ───────────────────────
# config_cpu/gpu.yaml ncore/kpar match dft_array/gpu_dft_array MPI layouts.
if [[ -n "${GPU_DFT}" ]]; then
    DFT_CONFIG="${SCRIPT_ROOT}/dft/config_gpu.yaml"
    DFT_ARRAY_SLURM="${SCRIPT_ROOT}/slurm/example/gpu_dft_array.slurm"
else
    DFT_CONFIG="${SCRIPT_ROOT}/dft/config_cpu.yaml"
    DFT_ARRAY_SLURM="${SCRIPT_ROOT}/slurm/example/dft_array.slurm"
fi
python "${SCRIPT_ROOT}/dft/prepare_dft_inputs.py" \
    --subset "${RUN_NAME}/diverse_subset.xyz" \
    --output "${RUN_NAME}/dft_inputs" \
    --config "${DFT_CONFIG}"

# ── Submit DFT array ─────────────────────────────────────────────────────────
N_FRAMES=$(find "${SIM_DIR}/${RUN_NAME}/dft_inputs" -maxdepth 1 -name 'frame_????' -type d | wc -l)
echo "DFT frames to submit: ${N_FRAMES}  (GPU_DFT=${GPU_DFT:-0} -> ${DFT_ARRAY_SLURM})"

DFT_JOB=$(sbatch \
    --parsable \
    --array=0-$((N_FRAMES - 1)) \
    --export=ALL,DFT_INPUT_DIR="${SIM_DIR}/${RUN_NAME}/dft_inputs",VENV_PATH="${VENV_PATH}" \
    "${DFT_ARRAY_SLURM}")
echo "DFT array job: ${DFT_JOB}"

# ── Submit Phase 2 (starts after all DFT tasks finish) ────────────────────────
sbatch \
    --dependency=afterok:"${DFT_JOB}" \
    --export=ALL,ITER="${ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${MODEL_NAME}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",NSTEPS="${NSTEPS}",N_RUN="${N_RUN}",N_FROZEN="${N_FROZEN}",COMMITTEE_SIZE="${COMMITTEE_SIZE}",PRIOR_WEIGHT="${PRIOR_WEIGHT}",ENERGY_WEIGHT="${ENERGY_WEIGHT}",FORCES_WEIGHT="${FORCES_WEIGHT}",BATCH_SIZE="${BATCH_SIZE}",SEED="${SEED}",TAU_REL="${TAU_REL}",TAU_HIST="${TAU_HIST}",SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD}",SELECTOR_DELAY="${SELECTOR_DELAY}",SELECTOR_MIXING="${SELECTOR_MIXING}",SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL}",SELECTOR_INTERVAL="${SELECTOR_INTERVAL}",T_VAL="${T_VAL}",FRICTION="${FRICTION}",TIMESTEP_FS="${TIMESTEP_FS}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}",GPU_DFT="${GPU_DFT}" \
    "${SCRIPT_ROOT}/slurm/example/uhal_phase2_collect_train.slurm"

echo "UHAL Phase 1 complete.  DFT job ${DFT_JOB} queued; Phase 2 queued with dependency."
