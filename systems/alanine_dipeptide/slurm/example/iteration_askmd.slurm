#!/bin/bash
#SBATCH --job-name=ad_al_askmd_it
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=01:30:00

# EXAMPLE SLURM script (Quadro RTX 6000 GPU, 4 cores): adapt the #SBATCH lines to your site and set
# SKMD_MODULES to the modules your builds need (README, "Cluster environment").

# a-SKMD (Eq. 8 online acquisition) variant of iteration.slurm. Eq. 8 already selects, so only
# label_selected_data.py runs (no FPS); --target_samples may stop early, so the checkpoint is found dynamically.

# ── Environment ──────────────────────────────────────────────────────────────
# Optional: SKMD_MODULES = space-separated module list to load.
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"          # .../alanine_dipeptide/askmd
: "${FILE_ROOT:?FILE_ROOT not set}"      # .../alanine_dipeptide (data root)
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"  # .../systems/alanine_dipeptide
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-100}"

RUN_NAME="${RUN_NAME:-askmd_a_3_T_50}"
SEED="${SEED:-123456}"

NSTEPS="${NSTEPS:-300}"
RELAX_START="${RELAX_START:-50}"
N_RUN="${N_RUN:-500}"
N_SILENT_BURNIN="${N_SILENT_BURNIN:-4000}"
N_BURNIN_MD="${N_BURNIN_MD:-6000}"
EVERY_BURNIN="${EVERY_BURNIN:-200}"
A_VAL="${A_VAL:-3.0}"
B_VAL="${B_VAL:-0.7}"
B_MODE="${B_MODE:-adaptive-median}"
B_MAX_MULT="${B_MAX_MULT:-8.0}"
T_VAL="${T_VAL:-50.0}"
DAMPING="${DAMPING:-75.0}"
DYN_TYPE="${DYN_TYPE:-overdamped}"
THRESH_PERCENTILE="${THRESH_PERCENTILE:-10.0}"
THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW:-20}"
TARGET_SAMPLES="${TARGET_SAMPLES:-100}"
BUFFER="${BUFFER:-250}"

LAMMPS_DATA="${FILE_ROOT}/ace-ala-nme.data"
INIT_XYZ="${FILE_ROOT}/ace-ala-nme.xyz"
NPZ="${FILE_ROOT}/phi_psi_relaxed_pes_off23_small.npz"
ORACLE_PATH="${FILE_ROOT}/MACE-OFF23_small.model"
ATOMIC_NUMBERS=(1 6 7 8)

SIM_DIR="${AL_ROOT}/simulation/iteration_${ITER}"
TRAIN_DIR="${AL_ROOT}/training/iteration_${ITER}"
mkdir -p "${SIM_DIR}" "${TRAIN_DIR}"
cd "${SIM_DIR}"

echo "=== a-SKMD AL iteration ${ITER} / ${N_ITERS} ==="
echo "Model: ${MODEL_NAME}   target_samples: ${TARGET_SAMPLES}   relax_start: ${RELAX_START}"

# Locate the foundation model for this iteration.
if [[ "${ITER}" -eq 1 ]]; then
    MODEL_SRC="${FILE_ROOT}/${MODEL_NAME}.model"
else
    MODEL_SRC="${AL_ROOT}/training/iteration_$((ITER - 1))/${MODEL_NAME}.model"
fi
cp "${MODEL_SRC}" .

symmetrix_extract_mace -m "${MODEL_NAME}.model" --atomic-numbers "${ATOMIC_NUMBERS[@]}"
MODEL_JSON="${MODEL_NAME}-1-6-7-8.json"

RUN_ARGS=(
    --lammps_data "${LAMMPS_DATA}"
    --init_xyz "${INIT_XYZ}"
    --model_json "${MODEL_JSON}"
    --fname "${RUN_NAME}"
    --nsteps "${NSTEPS}"
    --relax_start "${RELAX_START}"
    --npz_path "${NPZ}"
    --a "${A_VAL}"
    --b "${B_VAL}"
    --b_mode "${B_MODE}"
    --b_max_mult "${B_MAX_MULT}"
    --damping "${DAMPING}"
    --T "${T_VAL}"
    --n_run "${N_RUN}"
    --n_silent_burnin "${N_SILENT_BURNIN}"
    --n_burnin_md "${N_BURNIN_MD}"
    --every_burnin "${EVERY_BURNIN}"
    --dyn_type "${DYN_TYPE}"
    --seed "${SEED}"
    --thresh_percentile "${THRESH_PERCENTILE}"
    --thresh_calib_window "${THRESH_CALIB_WINDOW}"
    --target_samples "${TARGET_SAMPLES}"
    --buffer "${BUFFER}"
    --kokkos
    --kokkos_gpus 1
)

LABEL_ARGS=(
    --fname "${RUN_NAME}"
    --oracle_model "${ORACLE_PATH}"
    --npz_path "${NPZ}"
)

if [[ "${ITER}" -ge 2 ]]; then
    PREV_SIM_DIR="${AL_ROOT}/simulation/iteration_$((ITER - 1))/${RUN_NAME}"
    # Latest checkpoint: --target_samples may have stopped the previous loop early.
    PREV_CHECKPOINT_PATH=$(ls -v "${PREV_SIM_DIR}/traj_data"/configs_before_it_*.xyz 2>/dev/null | tail -1)
    [[ -n "${PREV_CHECKPOINT_PATH}" ]] || {
        echo "ERROR: no configs_before_it_*.xyz checkpoint found in ${PREV_SIM_DIR}/traj_data" >&2
        exit 1
    }
    INIT_POS_BASENAME="$(basename "${PREV_CHECKPOINT_PATH}")"
    cp "${PREV_CHECKPOINT_PATH}" .
    RUN_ARGS+=(--initial_particle_positions "${INIT_POS_BASENAME}")

    cp "${AL_ROOT}/training/iteration_$((ITER - 1))/training_data_it_$((ITER - 1)).xyz" prev_samples.xyz
    LABEL_ARGS+=(--prev_samples prev_samples.xyz)
fi

# ── SKMD exploration MD ───────────────────────────────────────────────────────
# Tolerate the harmless teardown crash (see iteration.slurm); completion marker is selected_data.xyz.
COMPLETION_MARKER="${SIM_DIR}/${RUN_NAME}/selected_data.xyz"
if [[ -n "${SKIP_MD:-}" ]]; then
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

# ── Oracle-label the online-selected data (no FPS -- Eq 8 already selected) ───
python "${SCRIPT_ROOT}/skmd_run/label_selected_data.py" "${LABEL_ARGS[@]}"

# ── Assemble this iteration's training set ─────────────────────────────────────
cd "${TRAIN_DIR}"
if [[ "${ITER}" -eq 1 ]]; then
    cp "${SIM_DIR}/${RUN_NAME}/diverse_subset.xyz" "training_data_it_${ITER}.xyz"
else
    cp "${SIM_DIR}/${RUN_NAME}/new_and_previous_samples.xyz" "training_data_it_${ITER}.xyz"
fi

# ── LoRA fine-tuning ────────────────────────────────────────────────────────────
cp "${SIM_DIR}/${MODEL_NAME}.model" .

NEW_MODEL="mace_off_32_askmd_it_${ITER}"

mace_run_train \
    --config           "${SCRIPT_ROOT}/training/lora_input.yaml" \
    --name             "${NEW_MODEL}_trained" \
    --train_file       "training_data_it_${ITER}.xyz" \
    --foundation_model "${MODEL_NAME}.model"

mv "${NEW_MODEL}_trained_stagetwo.model" "${NEW_MODEL}.model"
echo "Model: ${TRAIN_DIR}/${NEW_MODEL}.model"

symmetrix_extract_mace -m "${NEW_MODEL}.model" --atomic-numbers "${ATOMIC_NUMBERS[@]}"
echo "JSON:  ${TRAIN_DIR}/${NEW_MODEL}-1-6-7-8.json"

# ── Evaluate on this iteration's training set ──────────────────────────────────
python "${SCRIPT_ROOT}/training/evaluate_model.py" \
    --model    "${TRAIN_DIR}/${NEW_MODEL}.model" \
    --test-xyz "${TRAIN_DIR}/training_data_it_${ITER}.xyz" \
    --output   "${TRAIN_DIR}/train_metrics.json"

if [[ -f "${FILE_ROOT}/test_set.xyz" ]]; then
    python "${SCRIPT_ROOT}/training/evaluate_model.py" \
        --model    "${TRAIN_DIR}/${NEW_MODEL}.model" \
        --test-xyz "${FILE_ROOT}/test_set.xyz" \
        --output   "${TRAIN_DIR}/test_metrics.json"
else
    echo "[iteration] no ${FILE_ROOT}/test_set.xyz found; skipping held-out evaluation"
fi

# ── Propagate to next iteration ─────────────────────────────────────────────────
if [[ "${ITER}" -lt "${N_ITERS}" ]]; then
    NEXT_ITER=$((ITER + 1))
    mkdir -p "${AL_ROOT}/simulation/iteration_${NEXT_ITER}" "${AL_ROOT}/training/iteration_${NEXT_ITER}"

    sbatch \
        --export=ALL,ITER="${NEXT_ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${NEW_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",DYN_TYPE="${DYN_TYPE}",THRESH_PERCENTILE="${THRESH_PERCENTILE}",THRESH_CALIB_WINDOW="${THRESH_CALIB_WINDOW}",TARGET_SAMPLES="${TARGET_SAMPLES}",BUFFER="${BUFFER}" \
        "${SCRIPT_ROOT}/slurm/example/iteration_askmd.slurm"
    echo "Submitted iteration ${NEXT_ITER}."
else
    echo "All ${N_ITERS} iterations complete."
fi
