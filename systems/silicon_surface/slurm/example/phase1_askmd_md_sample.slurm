#!/bin/bash
#SBATCH --job-name=si_al_askmd_p1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:00:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# a-SKMD (Eq. 8 online acquisition) variant of phase1_md_sample.slurm: run_skmd_mace.py writes
# selected_data.xyz directly (no FPS); ITER>=2 restarts from the latest configs_before_it_*.xyz.

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"          # .../silicon_surface/askmd
: "${FILE_ROOT:?FILE_ROOT not set}"      # .../silicon_surface
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"  # .../systems/silicon_surface
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-30}"

RUN_NAME="${RUN_NAME:-askmd}"
SEED="${SEED:-123456}"

# Same dynamics as offline SKMD; only online acquisition (--target_samples/--thresh) differs.
NSTEPS="${NSTEPS:-200}"
RELAX_START="${RELAX_START:-50}"
N_RUN="${N_RUN:-500}"
A_VAL="${A_VAL:-4.0}"
B_VAL="${B_VAL:-1.0}"
B_MODE="${B_MODE:-adaptive-median}"
B_MAX_MULT="${B_MAX_MULT:-15.0}"
T_VAL="${T_VAL:-50.0}"
DAMPING="${DAMPING:-75.0}"
DYN_TYPE="${DYN_TYPE:-overdamped}"
N_FROZEN="${N_FROZEN:?N_FROZEN not set — see prepare_lammps_data.py}"

# Online acquisition (see run_skmd_mace.py --help); TARGET_SAMPLES defaults to N_SELECT (matched budget).
TARGET_SAMPLES="${TARGET_SAMPLES:-${N_SELECT}}"
THRESH_PERCENTILE="${THRESH_PERCENTILE:-10.0}"
THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW:-20}"
BUFFER="${BUFFER:-250}"

GPU_DFT="${GPU_DFT:-1}"

LAMMPS_DATA="${FILE_ROOT}/silicon_slab.data"
INIT_XYZ="${FILE_ROOT}/relaxed_struct_non_pandey.xyz"

SIM_DIR="${AL_ROOT}/simulation/iteration_${ITER}"
mkdir -p "${SIM_DIR}"
cd "${SIM_DIR}"

echo "=== a-SKMD Phase 1: iteration ${ITER} / ${N_ITERS} ==="
echo "Model: ${MODEL_NAME}   target_samples: ${TARGET_SAMPLES}"

# Locate the foundation model for this iteration.
if [[ "${ITER}" -eq 1 ]]; then
    MODEL_SRC="${FILE_ROOT}/${MODEL_NAME}.model"
else
    MODEL_SRC="${AL_ROOT}/training/iteration_$((ITER - 1))/${MODEL_NAME}.model"
fi
cp "${MODEL_SRC}" .

# Export LAMMPS JSON (symmetrix/mace pair style needs this). Si = Z 14.
symmetrix_extract_mace -m "${MODEL_NAME}.model" --atomic-numbers 14
MODEL_JSON="${MODEL_NAME}-14.json"

RUN_ARGS=(
    --lammps_data "${LAMMPS_DATA}"
    --init_xyz "${INIT_XYZ}"
    --model_json "${MODEL_JSON}"
    --elements Si
    --n_frozen "${N_FROZEN}"
    --fname "${RUN_NAME}"
    --nsteps "${NSTEPS}"
    --relax_start "${RELAX_START}"
    --a "${A_VAL}"
    --b "${B_VAL}"
    --b_mode "${B_MODE}"
    --b_max_mult "${B_MAX_MULT}"
    --damping "${DAMPING}"
    --T "${T_VAL}"
    --n_run "${N_RUN}"
    --dyn_type "${DYN_TYPE}"
    --seed "${SEED}"
    --target_samples "${TARGET_SAMPLES}"
    --thresh_percentile "${THRESH_PERCENTILE}"
    --thresh_calib_window "${THRESH_CALIB_WINDOW}"
    --buffer "${BUFFER}"
    --kokkos
    --kokkos_gpus 1
)

if [[ "${ITER}" -ge 2 ]]; then
    PREV_SIM_DIR="${AL_ROOT}/simulation/iteration_$((ITER - 1))/${RUN_NAME}"
    PREV_CHECKPOINT_PATH=$(ls -v "${PREV_SIM_DIR}/traj_data"/configs_before_it_*.xyz 2>/dev/null | tail -1)
    if [[ -z "${PREV_CHECKPOINT_PATH}" ]]; then
        echo "ERROR: no configs_before_it_*.xyz checkpoint found under ${PREV_SIM_DIR}/traj_data" >&2
        exit 1
    fi
    INIT_POS_BASENAME="$(basename "${PREV_CHECKPOINT_PATH}")"
    cp "${PREV_CHECKPOINT_PATH}" .
    RUN_ARGS+=(--initial_particle_positions "${INIT_POS_BASENAME}")

    cp "${AL_ROOT}/training/iteration_$((ITER - 1))/training_data_it_$((ITER - 1)).xyz" prev_samples.xyz
fi

# ── a-SKMD exploration MD (online selection; teardown-crash handling as in phase1_md_sample) ──
COMPLETION_MARKER="${SIM_DIR}/${RUN_NAME}/selected_data.xyz"
if [[ -n "${SKIP_MD:-}" ]]; then
    # SKIP_MD=1: MD already completed (see phase1_md_sample.slurm).
    [[ -f "${COMPLETION_MARKER}" ]] || {
        echo "SKIP_MD set but ${COMPLETION_MARKER} not found — nothing to skip to, aborting" >&2
        exit 1
    }
    echo "SKIP_MD set and ${COMPLETION_MARKER} exists — skipping run_skmd_mace.py"
    unset SKIP_MD
else
    set +e
    python "${SCRIPT_ROOT}/skmd_run/run_skmd_mace.py" "${RUN_ARGS[@]}"
    RUN_SKMD_EXIT=$?
    set -e
    if [[ "${RUN_SKMD_EXIT}" -ne 0 ]]; then
        if [[ -f "${COMPLETION_MARKER}" ]]; then
            echo "run_skmd_mace.py exited nonzero (${RUN_SKMD_EXIT}) but ${COMPLETION_MARKER} exists — known harmless LAMMPS/symmetrix teardown crash after completing, continuing"
        else
            echo "run_skmd_mace.py failed with exit code ${RUN_SKMD_EXIT} (no completion marker at ${COMPLETION_MARKER})" >&2
            exit "${RUN_SKMD_EXIT}"
        fi
    fi
fi

N_COLLECTED=$(python -c "from ase.io import read; print(len(read('${RUN_NAME}/selected_data.xyz', index=':')))" 2>/dev/null || echo 0)
echo "a-SKMD collected ${N_COLLECTED} structure(s) online (target was ${TARGET_SAMPLES})"

# ── Write VASP inputs directly from the online selection ──────────────────────
# config_cpu/gpu.yaml ncore/kpar match dft_array/gpu_dft_array MPI layouts.
if [[ -n "${GPU_DFT}" ]]; then
    DFT_CONFIG="${SCRIPT_ROOT}/dft/config_gpu.yaml"
    DFT_ARRAY_SLURM="${SCRIPT_ROOT}/slurm/example/gpu_dft_array.slurm"
else
    DFT_CONFIG="${SCRIPT_ROOT}/dft/config_cpu.yaml"
    DFT_ARRAY_SLURM="${SCRIPT_ROOT}/slurm/example/dft_array.slurm"
fi
python "${SCRIPT_ROOT}/dft/prepare_dft_inputs.py" \
    --subset "${RUN_NAME}/selected_data.xyz" \
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
    --export=ALL,ITER="${ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${MODEL_NAME}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",N_FROZEN="${N_FROZEN}",DYN_TYPE="${DYN_TYPE}",TARGET_SAMPLES="${TARGET_SAMPLES}",THRESH_PERCENTILE="${THRESH_PERCENTILE}",THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW}",BUFFER="${BUFFER}",GPU_DFT="${GPU_DFT}" \
    "${SCRIPT_ROOT}/slurm/example/phase2_askmd_collect_train.slurm"

echo "a-SKMD Phase 1 complete.  DFT job ${DFT_JOB} queued; Phase 2 queued with dependency."
