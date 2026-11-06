#!/usr/bin/env bash
# AL loop driver for alanine dipeptide (MACE-OFF23-small oracle, offline FPS), shared by the offline
# SKMD arm, the 50 K baseline (--a 0) and the Langevin arms (--dyn-type underdamped --a 0).
# Submits iteration 1; slurm/example/iteration.slurm resubmits itself until NUM_ITERS.
#
# Usage:
#   ./run_al_loop.sh NUM_ITERS [--start-from K] [--n-select N] [--venv PATH] [--dyn-type TYPE] [--skip-md]
#                              [--a VAL] [--seed N] [--run-name NAME] [--al-subdir NAME]
#                              [--a_relax_mult V] [--b_relax_mult V] [--T VAL] [--damping VAL]
#
# Examples:
#   ./run_al_loop.sh 5
#   ./run_al_loop.sh 5 --start-from 3   # resume; iters 1-2 must be complete
#   ./run_al_loop.sh 5 --n-select 50
#   ./run_al_loop.sh 5 --dyn-type underdamped   # Langevin instead of Brownian dynamics
#   ./run_al_loop.sh 5 --start-from 3 --skip-md # iter 3 MD done; resume at post-processing
#   ./run_al_loop.sh 5 --a 0 --seed 1 --run-name unbiased_a_0_T_50_seed_1 \
#       --al-subdir baseline_seed_1              # overdamped 50 K baseline (bias removed)

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FILE_ROOT="$(cd "${SCRIPT_ROOT}/../../data/alanine_dipeptide" && pwd)"
AL_SUBDIR="skmd"

NUM_ITERS=""
START_FROM=1
N_SELECT=100
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)
SKIP_MD=""

# SKMD hyperparameters (paper App. E.2): 300 outer steps x 500 MD steps, median-bandwidth
# exploration until step 250, a = 3.0, overdamped 50 K, damping 75.
NSTEPS=300
RELAX_START=250
N_RUN=500
A_VAL=3.0
B_VAL=0.7
B_MODE=adaptive
B_MAX_MULT=3.0
T_VAL=50.0
DAMPING=75.0
DYN_TYPE=overdamped
SEED=123456
RUN_NAME="overdamped_a_3_T_50"

THRESH="${THRESH:-}"
BUFFER="${BUFFER:-}"

B_STABILITY_TOL="${B_STABILITY_TOL:-}"
B_STABILITY_WINDOW="${B_STABILITY_WINDOW:-}"

# Relaxation-phase a/b decay multipliers (run_skmd_mace.py defaults); 1.0 holds a/b constant post-freeze.
A_RELAX_MULT="${A_RELAX_MULT:-0.5}"
B_RELAX_MULT="${B_RELAX_MULT:-0.5}"

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
        --dyn-type)
            DYN_TYPE="${2:?--dyn-type requires overdamped or underdamped}"
            shift 2 ;;
        --skip-md)
            SKIP_MD=1
            shift ;;
        --a)
            A_VAL="${2:?--a requires a number}"
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
        --a_relax_mult)
            A_RELAX_MULT="${2:?--a_relax_mult requires a number}"
            shift 2 ;;
        --b_relax_mult)
            B_RELAX_MULT="${2:?--b_relax_mult requires a number}"
            shift 2 ;;
        --T)
            T_VAL="${2:?--T requires a number}"
            shift 2 ;;
        --damping)
            DAMPING="${2:?--damping requires a number}"
            shift 2 ;;
        -h|--help)
            sed -n '2,19p' "$0" | sed 's/^# *//'
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
        echo "mace_off_32_skmd_it_${iter}"
    fi
}

model_path_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "${FILE_ROOT}/${BASE_MODEL_NAME}.model"
    else
        echo "${AL_ROOT}/training/iteration_${iter}/mace_off_32_skmd_it_${iter}.model"
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

echo "AL loop (SLURM): ${NUM_ITERS} iterations, starting from ${START_FROM}"
echo "Selections per iteration: ${N_SELECT}   DYN_TYPE: ${DYN_TYPE}   A_VAL: ${A_VAL}   SEED: ${SEED}"
echo "RUN_NAME: ${RUN_NAME}   AL_ROOT: ${AL_ROOT}"
echo "venv: ${VENV_PATH}"
if [[ -n "${SKIP_MD}" ]]; then
    echo "--skip-md set: iteration ${START_FROM}'s MD step will be skipped (iteration.slurm will abort if its completion marker isn't already there)"
fi

CURRENT_MODEL=$(model_name_for_iter "${PREV_ITER}")
echo "Foundation model: ${CURRENT_MODEL}"

ITER_JOB=$(sbatch \
    --parsable \
    --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",DYN_TYPE="${DYN_TYPE}",THRESH="${THRESH}",BUFFER="${BUFFER}",B_STABILITY_TOL="${B_STABILITY_TOL}",B_STABILITY_WINDOW="${B_STABILITY_WINDOW}",A_RELAX_MULT="${A_RELAX_MULT}",B_RELAX_MULT="${B_RELAX_MULT}",SKIP_MD="${SKIP_MD}" \
    "${SCRIPT_ROOT}/slurm/example/iteration.slurm")
echo "Submitted iteration ${START_FROM}: job ${ITER_JOB}"

echo "Chain will self-propagate through all ${NUM_ITERS} iterations automatically."
