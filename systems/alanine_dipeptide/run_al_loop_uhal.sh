#!/usr/bin/env bash
# UHAL (ase_uhal hyperactive learning) AL loop driver for alanine dipeptide: submits iteration 1 of
# the self-chaining slurm/example/iteration_uhal.slurm. Adapted from systems/silicon_surface.
#
# Usage:
#   ./run_al_loop_uhal.sh NUM_ITERS [--start-from K] [--n-select N] [--venv PATH]
#                         [--seed N] [--run-name NAME] [--al-subdir NAME]
#                         [--T VAL] [--friction VAL] [--committee_size N]
#
# Examples:
#   ./run_al_loop_uhal.sh 5
#   ./run_al_loop_uhal.sh 5 --start-from 3   # resume; iters 1-2 must be complete

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FILE_ROOT="$(cd "${SCRIPT_ROOT}/../../data/alanine_dipeptide" && pwd)"
AL_SUBDIR="uhal"

NUM_ITERS=""
START_FROM=1
N_SELECT=100
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)

# PRIOR_WEIGHT/FORCES_WEIGHT/TAU_REL differ from silicon's 0.1/100.0/0.1, at which tau blew up
# ~30x within 3 selections; values from pilot runs. NSTEPS must match the SKMD drivers (budget = nsteps*n_run).
NSTEPS=300
N_RUN=500
COMMITTEE_SIZE=10
PRIOR_WEIGHT=0.5
ENERGY_WEIGHT=1.0
FORCES_WEIGHT=10.0
BATCH_SIZE=8
SEED=42
TAU_REL=0.03
TAU_HIST=10
SELECTOR_THRESHOLD=adaptive
SELECTOR_DELAY=10
SELECTOR_MIXING=0.1
SELECTOR_THRESH_MUL=1.2
SELECTOR_INTERVAL=2
T_VAL=700.0
FRICTION=0.01
TIMESTEP_FS=1.0
# Burn-in yields N = n_burnin_md // every_burnin walkers (~30), matching a-SKMD's ensemble.
N_SILENT_BURNIN=4000
N_BURNIN_MD=6000
EVERY_BURNIN=200
RUN_NAME="uhal_T_700"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from)           START_FROM="${2:?--start-from requires an integer}"; shift 2 ;;
        --n-select)             N_SELECT="${2:?--n-select requires an integer}"; shift 2 ;;
        --venv)                 VENV_PATH="${2:?--venv requires a path}"; shift 2 ;;
        --seed)                 SEED="${2:?--seed requires an integer}"; shift 2 ;;
        --run-name)             RUN_NAME="${2:?--run-name requires a name}"; shift 2 ;;
        --al-subdir)            AL_SUBDIR="${2:?--al-subdir requires a name}"; shift 2 ;;
        --T)                    T_VAL="${2:?--T requires a number}"; shift 2 ;;
        --friction)             FRICTION="${2:?--friction requires a number}"; shift 2 ;;
        --committee_size)       COMMITTEE_SIZE="${2:?--committee_size requires an integer}"; shift 2 ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# *//'
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

BASE_MODEL_NAME="mace_off_32_0.1_percent"

model_name_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "${BASE_MODEL_NAME}"
    else
        echo "mace_off_32_uhal_it_${iter}"
    fi
}

model_path_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "${FILE_ROOT}/${BASE_MODEL_NAME}.model"
    else
        echo "${AL_ROOT}/training/iteration_${iter}/mace_off_32_uhal_it_${iter}.model"
    fi
}

# ── Prerequisite checks ─────────────────────────────────────────────────────────

PREV_ITER=$((START_FROM - 1))
PREV_MODEL_PATH=$(model_path_for_iter "${PREV_ITER}")
[[ -f "${PREV_MODEL_PATH}" ]] || {
    echo "ERROR: prerequisite model not found: ${PREV_MODEL_PATH}" >&2
    exit 1
}

[[ -f "${FILE_ROOT}/ace-ala-nme.xyz" ]] || {
    echo "ERROR: ${FILE_ROOT}/ace-ala-nme.xyz not found" >&2
    exit 1
}

# ── Setup runtime directories ────────────────────────────────────────────────

mkdir -p "${AL_ROOT}/simulation/iteration_${START_FROM}"
mkdir -p "${AL_ROOT}/training/iteration_${START_FROM}"

# ── Submit job ────────────────────────────────────────────────────────────────

echo "UHAL AL loop (SLURM): ${NUM_ITERS} iterations, starting from ${START_FROM}"
echo "Selections per iteration (n_select): ${N_SELECT}   T: ${T_VAL}K   SEED: ${SEED}"
echo "RUN_NAME: ${RUN_NAME}   AL_ROOT: ${AL_ROOT}"
echo "venv: ${VENV_PATH}"

CURRENT_MODEL=$(model_name_for_iter "${PREV_ITER}")
echo "Foundation model: ${CURRENT_MODEL}"

ITER_JOB=$(sbatch \
    --parsable \
    --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",N_RUN="${N_RUN}",COMMITTEE_SIZE="${COMMITTEE_SIZE}",PRIOR_WEIGHT="${PRIOR_WEIGHT}",ENERGY_WEIGHT="${ENERGY_WEIGHT}",FORCES_WEIGHT="${FORCES_WEIGHT}",BATCH_SIZE="${BATCH_SIZE}",TAU_REL="${TAU_REL}",TAU_HIST="${TAU_HIST}",SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD}",SELECTOR_DELAY="${SELECTOR_DELAY}",SELECTOR_MIXING="${SELECTOR_MIXING}",SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL}",SELECTOR_INTERVAL="${SELECTOR_INTERVAL}",T_VAL="${T_VAL}",FRICTION="${FRICTION}",TIMESTEP_FS="${TIMESTEP_FS}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}" \
    "${SCRIPT_ROOT}/slurm/example/iteration_uhal.slurm")
echo "Submitted iteration ${START_FROM}: job ${ITER_JOB}"

echo "Chain will self-propagate through all ${NUM_ITERS} iterations automatically."
