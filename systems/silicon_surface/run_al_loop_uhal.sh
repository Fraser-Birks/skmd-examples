#!/usr/bin/env bash
# uHAL (ase_uhal) AL loop on Si(111) (VASP/SCAN oracle); same chain and MD budget as the SKMD drivers.
# Usage: ./run_al_loop_uhal.sh NUM_ITERS [--start-from K] [--n-select N] [--venv PATH] [--seed N]
#          [--run-name NAME] [--al-subdir NAME] [--cpu-dft] [--skip-md] [--skip-dft]

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FILE_ROOT="$(cd "${SCRIPT_ROOT}/../../data/silicon_surface" && pwd)"
AL_SUBDIR="uhal"

NUM_ITERS=""
START_FROM=1
N_SELECT=30
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)
SEED=42
RUN_NAME="uhal"
GPU_DFT=1
SKIP_MD=""
SKIP_DFT=""

# uHAL hyperparameters (see run_uhal_mace.py --help); 200 x 500 MD steps matches the SKMD budget.
NSTEPS=200
N_RUN=500
COMMITTEE_SIZE=10
PRIOR_WEIGHT=0.5
ENERGY_WEIGHT=1.0
FORCES_WEIGHT=10.0
BATCH_SIZE=8
TAU_REL=0.03
TAU_HIST=10
SELECTOR_THRESHOLD=adaptive
SELECTOR_DELAY=10
SELECTOR_MIXING=0.1
SELECTOR_THRESH_MUL=1.2
SELECTOR_INTERVAL=2
# ASE Langevin T/friction; not unit-comparable to the LAMMPS T_VAL/DAMPING.
T_VAL=700.0
FRICTION=0.01
TIMESTEP_FS=1.0
# Burn-in yields N = N_BURNIN_MD // EVERY_BURNIN walkers, matching the SKMD ensemble.
N_SILENT_BURNIN=4000
N_BURNIN_MD=6000
EVERY_BURNIN=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from)
            START_FROM="${2:?--start-from requires an integer}"
            shift 2 ;;
        --n-select)
            N_SELECT="${2:?--n-select requires an integer}"
            shift 2 ;;
        --venv)
            VENV_PATH="${2:?--venv requires a path}"
            shift 2 ;;
        --seed)
            SEED="${2:?--seed requires an integer}"
            shift 2 ;;
        --run-name)
            RUN_NAME="${2:?--run-name requires a name}"
            shift 2 ;;
        --al-subdir)
            AL_SUBDIR="${2:?--al-subdir requires a name}"
            shift 2 ;;
        --gpu-dft)
            GPU_DFT=1
            shift ;;
        --cpu-dft)
            GPU_DFT=""
            shift ;;
        --skip-md)
            SKIP_MD=1
            shift ;;
        --skip-dft)
            SKIP_DFT=1
            shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "${NUM_ITERS}" ]]; then
                NUM_ITERS="$1"
            else
                echo "Unexpected argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

: "${NUM_ITERS:?Usage: $0 NUM_ITERS [--start-from K] [--n-select N] [--venv PATH]}"
[[ "${NUM_ITERS}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ITERS must be a positive integer" >&2; exit 1; }
[[ "${START_FROM}" =~ ^[1-9][0-9]*$ ]] || { echo "--start-from must be a positive integer" >&2; exit 1; }
[[ "${N_SELECT}" =~ ^[1-9][0-9]*$ ]] || { echo "--n-select must be a positive integer" >&2; exit 1; }
(( START_FROM <= NUM_ITERS )) || { echo "--start-from (${START_FROM}) > NUM_ITERS (${NUM_ITERS})" >&2; exit 1; }
: "${VENV_PATH:?set SKMD_VENV=/path/to/venv (or pass --venv PATH)}"
VENV_PATH="$(realpath "${VENV_PATH}")"
[[ -f "${VENV_PATH}/bin/activate" ]] || {
    echo "ERROR: no venv found at ${VENV_PATH} (expected ${VENV_PATH}/bin/activate)" >&2
    exit 1
}

AL_ROOT="${FILE_ROOT}/${AL_SUBDIR}"

# ── Model name helpers ─────────────────────────────────────────────────────────

model_name_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "mace-mpa-0-medium"
    else
        echo "si_slab_uhal_lora_it_${iter}"
    fi
}

model_path_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "${FILE_ROOT}/mace-mpa-0-medium.model"
    else
        echo "${AL_ROOT}/training/iteration_${iter}/si_slab_uhal_lora_it_${iter}.model"
    fi
}

# ── Prerequisite checks ─────────────────────────────────────────────────────────

PREV_ITER=$((START_FROM - 1))
PREV_MODEL_PATH=$(model_path_for_iter "${PREV_ITER}")
[[ -f "${PREV_MODEL_PATH}" ]] || {
    echo "ERROR: prerequisite model not found: ${PREV_MODEL_PATH}" >&2
    exit 1
}

[[ -f "${FILE_ROOT}/silicon_slab.data" ]] || {
    echo "ERROR: ${FILE_ROOT}/silicon_slab.data not found — run prepare_lammps_data.py first" >&2
    exit 1
}

# N_FROZEN: env var, else the sidecar written by prepare_lammps_data.py.
if [[ -n "${N_FROZEN:-}" ]]; then
    :
elif [[ -f "${FILE_ROOT}/silicon_slab.data.n_frozen" ]]; then
    N_FROZEN="$(cat "${FILE_ROOT}/silicon_slab.data.n_frozen")"
else
    echo "ERROR: N_FROZEN not set and no ${FILE_ROOT}/silicon_slab.data.n_frozen sidecar found." >&2
    echo "       Set N_FROZEN explicitly or re-run prepare_lammps_data.py." >&2
    exit 1
fi

# ── Setup runtime directories ────────────────────────────────────────────────

mkdir -p "${AL_ROOT}/simulation/iteration_${START_FROM}"
mkdir -p "${AL_ROOT}/training/iteration_${START_FROM}"

# ── Submit jobs ───────────────────────────────────────────────────────────────

echo "UHAL AL loop (SLURM): ${NUM_ITERS} iterations, starting from ${START_FROM}"
echo "AL_ROOT: ${AL_ROOT}   RUN_NAME: ${RUN_NAME}   SEED: ${SEED}"
echo "Selections per iteration: ${N_SELECT}   N_FROZEN: ${N_FROZEN}"
echo "venv: ${VENV_PATH}"
if [[ -n "${SKIP_DFT}" ]]; then
    echo "--skip-dft set: iteration ${START_FROM}'s MD and DFT are both skipped -- submitting Phase 2 directly (its own DFT-collection step will abort if OUTCARs aren't already there)"
elif [[ -n "${SKIP_MD}" ]]; then
    echo "--skip-md set: iteration ${START_FROM}'s MD step will be skipped (uhal_phase1_md_sample.slurm will abort if its completion marker isn't already there)"
fi

CURRENT_MODEL=$(model_name_for_iter "${PREV_ITER}")
echo "Foundation model: ${CURRENT_MODEL}"

if [[ -n "${SKIP_DFT}" ]]; then
    # Requires complete dft_inputs/frame_*/OUTCAR; collect_dft_results.py aborts if none are usable.
    PHASE2_JOB=$(sbatch \
        --parsable \
        --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",NSTEPS="${NSTEPS}",N_RUN="${N_RUN}",N_FROZEN="${N_FROZEN}",COMMITTEE_SIZE="${COMMITTEE_SIZE}",PRIOR_WEIGHT="${PRIOR_WEIGHT}",ENERGY_WEIGHT="${ENERGY_WEIGHT}",FORCES_WEIGHT="${FORCES_WEIGHT}",BATCH_SIZE="${BATCH_SIZE}",SEED="${SEED}",TAU_REL="${TAU_REL}",TAU_HIST="${TAU_HIST}",SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD}",SELECTOR_DELAY="${SELECTOR_DELAY}",SELECTOR_MIXING="${SELECTOR_MIXING}",SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL}",SELECTOR_INTERVAL="${SELECTOR_INTERVAL}",T_VAL="${T_VAL}",FRICTION="${FRICTION}",TIMESTEP_FS="${TIMESTEP_FS}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}",GPU_DFT="${GPU_DFT}" \
        "${SCRIPT_ROOT}/slurm/example/uhal_phase2_collect_train.slurm")
    echo "Submitted UHAL Phase 2 directly for iteration ${START_FROM}: job ${PHASE2_JOB}"
else
    PHASE1_JOB=$(sbatch \
        --parsable \
        --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",NSTEPS="${NSTEPS}",N_RUN="${N_RUN}",N_FROZEN="${N_FROZEN}",COMMITTEE_SIZE="${COMMITTEE_SIZE}",PRIOR_WEIGHT="${PRIOR_WEIGHT}",ENERGY_WEIGHT="${ENERGY_WEIGHT}",FORCES_WEIGHT="${FORCES_WEIGHT}",BATCH_SIZE="${BATCH_SIZE}",SEED="${SEED}",TAU_REL="${TAU_REL}",TAU_HIST="${TAU_HIST}",SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD}",SELECTOR_DELAY="${SELECTOR_DELAY}",SELECTOR_MIXING="${SELECTOR_MIXING}",SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL}",SELECTOR_INTERVAL="${SELECTOR_INTERVAL}",T_VAL="${T_VAL}",FRICTION="${FRICTION}",TIMESTEP_FS="${TIMESTEP_FS}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}",GPU_DFT="${GPU_DFT}",SKIP_MD="${SKIP_MD}" \
        "${SCRIPT_ROOT}/slurm/example/uhal_phase1_md_sample.slurm")
    echo "Submitted UHAL Phase 1 for iteration ${START_FROM}: job ${PHASE1_JOB}"
fi

echo "Chain will self-propagate through all ${NUM_ITERS} iterations automatically."
