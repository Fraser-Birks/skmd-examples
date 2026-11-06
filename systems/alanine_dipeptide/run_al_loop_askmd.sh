#!/usr/bin/env bash
# a-SKMD (Eq. 8 online acquisition) AL loop driver for alanine dipeptide: submits iteration 1 of
# the self-chaining slurm/example/iteration_askmd.slurm (online labelling, no offline FPS).
#
# Usage:
#   ./run_al_loop_askmd.sh NUM_ITERS [--start-from K] [--n-select N] [--venv PATH] [--dyn-type TYPE]
#                          [--seed N] [--run-name NAME] [--al-subdir NAME]
#                          [--relax_start N] [--b_max_mult V] [--target_samples N]
#                          [--buffer N] [--thresh_percentile V] [--thresh_calib_window N]
#
# Examples:
#   ./run_al_loop_askmd.sh 5
#   ./run_al_loop_askmd.sh 5 --start-from 3   # resume; iters 1-2 must be complete

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FILE_ROOT="$(cd "${SCRIPT_ROOT}/../../data/alanine_dipeptide" && pwd)"
AL_SUBDIR="askmd"

NUM_ITERS=""
START_FROM=1
N_SELECT=100
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)
DYN_TYPE=overdamped

# a-SKMD hyperparameters (paper App. E.2): 50 exploration steps of 300; zeta_0 = 10th percentile
# of force norms over the last 20 exploration steps; min l_0 = 250 MD steps before stopping check.
NSTEPS=300
RELAX_START=50
N_RUN=500
N_SILENT_BURNIN=4000
N_BURNIN_MD=6000
EVERY_BURNIN=200
A_VAL=3.0
B_VAL=0.7
B_MODE=adaptive-median
B_MAX_MULT=8.0
T_VAL=50.0
DAMPING=75.0
SEED=123456
RUN_NAME="askmd_a_3_T_50"
THRESH_PERCENTILE=10.0
THRESH_CALIB_WINDOW=20
TARGET_SAMPLES=100
BUFFER=250

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from)           START_FROM="${2:?--start-from requires an integer}"; shift 2 ;;
        --n-select)             N_SELECT="${2:?--n-select requires an integer}"; shift 2 ;;
        --venv)                 VENV_PATH="${2:?--venv requires a path}"; shift 2 ;;
        --dyn-type)             DYN_TYPE="${2:?--dyn-type requires overdamped or underdamped}"; shift 2 ;;
        --a)                    A_VAL="${2:?--a requires a number}"; shift 2 ;;
        --seed)                 SEED="${2:?--seed requires an integer}"; shift 2 ;;
        --run-name)             RUN_NAME="${2:?--run-name requires a name}"; shift 2 ;;
        --al-subdir)            AL_SUBDIR="${2:?--al-subdir requires a name}"; shift 2 ;;
        --relax_start)          RELAX_START="${2:?--relax_start requires an integer}"; shift 2 ;;
        --b_max_mult)           B_MAX_MULT="${2:?--b_max_mult requires a number}"; shift 2 ;;
        --target_samples)       TARGET_SAMPLES="${2:?--target_samples requires an integer}"; shift 2 ;;
        --buffer)                BUFFER="${2:?--buffer requires an integer}";                 shift 2 ;;
        --thresh_percentile)    THRESH_PERCENTILE="${2:?--thresh_percentile requires a number}"; shift 2 ;;
        --thresh_calib_window)  THRESH_CALIB_WINDOW="${2:?--thresh_calib_window requires an integer}"; shift 2 ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# *//'
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
[[ "${DYN_TYPE}" == "overdamped" || "${DYN_TYPE}" == "underdamped" ]] || {
    echo "--dyn-type must be 'overdamped' or 'underdamped' (got '${DYN_TYPE}')" >&2
    exit 1
}
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
        echo "mace_off_32_askmd_it_${iter}"
    fi
}

model_path_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "${FILE_ROOT}/${BASE_MODEL_NAME}.model"
    else
        echo "${AL_ROOT}/training/iteration_${iter}/mace_off_32_askmd_it_${iter}.model"
    fi
}

# ── Prerequisite checks ─────────────────────────────────────────────────────────

PREV_ITER=$((START_FROM - 1))
PREV_MODEL_PATH=$(model_path_for_iter "${PREV_ITER}")
[[ -f "${PREV_MODEL_PATH}" ]] || {
    echo "ERROR: prerequisite model not found: ${PREV_MODEL_PATH}" >&2
    exit 1
}

[[ -f "${FILE_ROOT}/ace-ala-nme.data" ]] || {
    echo "ERROR: ${FILE_ROOT}/ace-ala-nme.data not found" >&2
    exit 1
}

# ── Setup runtime directories ────────────────────────────────────────────────

mkdir -p "${AL_ROOT}/simulation/iteration_${START_FROM}"
mkdir -p "${AL_ROOT}/training/iteration_${START_FROM}"

# ── Submit job ────────────────────────────────────────────────────────────────

echo "a-SKMD AL loop (SLURM): ${NUM_ITERS} iterations, starting from ${START_FROM}"
echo "Selections per iteration (target_samples): ${TARGET_SAMPLES}   relax_start: ${RELAX_START}   SEED: ${SEED}"
echo "RUN_NAME: ${RUN_NAME}   AL_ROOT: ${AL_ROOT}"
echo "venv: ${VENV_PATH}"

CURRENT_MODEL=$(model_name_for_iter "${PREV_ITER}")
echo "Foundation model: ${CURRENT_MODEL}"

ITER_JOB=$(sbatch \
    --parsable \
    --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",DYN_TYPE="${DYN_TYPE}",THRESH_PERCENTILE="${THRESH_PERCENTILE}",THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW}",TARGET_SAMPLES="${TARGET_SAMPLES}",BUFFER="${BUFFER}" \
    "${SCRIPT_ROOT}/slurm/example/iteration_askmd.slurm")
echo "Submitted iteration ${START_FROM}: job ${ITER_JOB}"

echo "Chain will self-propagate through all ${NUM_ITERS} iterations automatically."
