#!/usr/bin/env bash
# a-SKMD active learning loop on Si(111) (VASP/SCAN oracle); online acquisition, defaults per paper App. E.3.
# Usage: ./run_al_loop_askmd.sh NUM_ITERS [--start-from K] [--n-select N] [--venv PATH] [--seed N]
#          [--run-name NAME] [--al-subdir NAME] [--skip-md] [--skip-dft] [--cpu-dft]

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FILE_ROOT="$(cd "${SCRIPT_ROOT}/../../data/silicon_surface" && pwd)"
AL_SUBDIR="askmd"

NUM_ITERS=""
START_FROM=1
N_SELECT=30
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)
SKIP_MD=""
SKIP_DFT=""
GPU_DFT=1
SEED=123456
RUN_NAME="askmd"

# Offline-SKMD dynamics; RELAX_START=50 so online acquisition mostly samples after early exploration.
NSTEPS=200
RELAX_START=50
N_RUN=500
A_VAL=4.0
B_VAL=1.0
B_MODE=adaptive-median
B_MAX_MULT=15.0
T_VAL=50.0
DAMPING=75.0
DYN_TYPE=overdamped

# Online acquisition (see run_skmd_mace.py --help); TARGET_SAMPLES defaults to N_SELECT.
TARGET_SAMPLES="${TARGET_SAMPLES:-}"
THRESH_PERCENTILE="${THRESH_PERCENTILE:-10.0}"
THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW:-20}"
BUFFER="${BUFFER:-250}"

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
        --skip-md)
            SKIP_MD=1
            shift ;;
        --skip-dft)
            SKIP_DFT=1
            shift ;;
        --gpu-dft)
            GPU_DFT=1
            shift ;;
        --cpu-dft)
            GPU_DFT=""
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
        echo "si_slab_askmd_lora_it_${iter}"
    fi
}

model_path_for_iter() {
    local iter="$1"
    if [[ "$iter" -eq 0 ]]; then
        echo "${FILE_ROOT}/mace-mpa-0-medium.model"
    else
        echo "${AL_ROOT}/training/iteration_${iter}/si_slab_askmd_lora_it_${iter}.model"
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

echo "a-SKMD AL loop (SLURM): ${NUM_ITERS} iterations, starting from ${START_FROM}"
echo "AL_ROOT: ${AL_ROOT}   RUN_NAME: ${RUN_NAME}   SEED: ${SEED}"
echo "n-select (target_samples default): ${N_SELECT}   N_FROZEN: ${N_FROZEN}"
echo "venv: ${VENV_PATH}"
if [[ -n "${SKIP_DFT}" ]]; then
    echo "--skip-dft set: iteration ${START_FROM}'s MD and DFT are both skipped -- submitting Phase 2 directly (its own DFT-collection step will abort if OUTCARs aren't already there)"
elif [[ -n "${SKIP_MD}" ]]; then
    echo "--skip-md set: iteration ${START_FROM}'s MD step will be skipped"
fi
if [[ -n "${GPU_DFT}" ]]; then
    echo "--gpu-dft set: DFT array jobs will run on GPU-built VASP (gpu_dft_array.slurm), every iteration"
fi

CURRENT_MODEL=$(model_name_for_iter "${PREV_ITER}")
echo "Foundation model: ${CURRENT_MODEL}"

if [[ -n "${SKIP_DFT}" ]]; then
    PHASE2_JOB=$(sbatch \
        --parsable \
        --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",N_FROZEN="${N_FROZEN}",DYN_TYPE="${DYN_TYPE}",TARGET_SAMPLES="${TARGET_SAMPLES}",THRESH_PERCENTILE="${THRESH_PERCENTILE}",THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW}",BUFFER="${BUFFER}",GPU_DFT="${GPU_DFT}" \
        "${SCRIPT_ROOT}/slurm/example/phase2_askmd_collect_train.slurm")
    echo "Submitted a-SKMD Phase 2 directly for iteration ${START_FROM}: job ${PHASE2_JOB}"
else
    PHASE1_JOB=$(sbatch \
        --parsable \
        --export=ALL,ITER="${START_FROM}",N_ITERS="${NUM_ITERS}",MODEL_NAME="${CURRENT_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",N_FROZEN="${N_FROZEN}",DYN_TYPE="${DYN_TYPE}",TARGET_SAMPLES="${TARGET_SAMPLES}",THRESH_PERCENTILE="${THRESH_PERCENTILE}",THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW}",BUFFER="${BUFFER}",SKIP_MD="${SKIP_MD}",GPU_DFT="${GPU_DFT}" \
        "${SCRIPT_ROOT}/slurm/example/phase1_askmd_md_sample.slurm")
    echo "Submitted a-SKMD Phase 1 for iteration ${START_FROM}: job ${PHASE1_JOB}"
fi

echo "Chain will self-propagate through all ${NUM_ITERS} iterations automatically."
