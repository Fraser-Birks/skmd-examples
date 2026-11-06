#!/bin/bash
#SBATCH --job-name=ad_al_uhal_it
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
# 4h: ASE MD at ~20 steps/s x 150000 steps is ~2h before training (LAMMPS SKMD needs 1h30).
#SBATCH --time=04:00:00

# EXAMPLE SLURM script (Quadro RTX 6000 GPU, 4 cores): adapt the #SBATCH lines to your site and set
# SKMD_MODULES to the modules your builds need (README, "Cluster environment").

# UHAL (ase_uhal) variant of iteration_askmd.slurm: plain .model (no symmetrix JSON), fresh burn-in
# from INIT_XYZ each iteration, online HAL selection so only label_selected_data.py runs (no FPS).

# ── Environment ──────────────────────────────────────────────────────────────
# Optional: SKMD_MODULES = space-separated module list to load.
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"          # .../alanine_dipeptide/uhal
: "${FILE_ROOT:?FILE_ROOT not set}"      # .../alanine_dipeptide (data root)
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"  # .../systems/alanine_dipeptide
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-100}"

RUN_NAME="${RUN_NAME:-uhal_T_700}"
SEED="${SEED:-42}"

NSTEPS="${NSTEPS:-300}"
N_RUN="${N_RUN:-500}"
COMMITTEE_SIZE="${COMMITTEE_SIZE:-10}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.5}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-10.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
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
N_SILENT_BURNIN="${N_SILENT_BURNIN:-4000}"
N_BURNIN_MD="${N_BURNIN_MD:-6000}"
EVERY_BURNIN="${EVERY_BURNIN:-200}"

INIT_XYZ="${FILE_ROOT}/ace-ala-nme.xyz"
NPZ="${FILE_ROOT}/phi_psi_relaxed_pes_off23_small.npz"
ORACLE_PATH="${FILE_ROOT}/MACE-OFF23_small.model"

SIM_DIR="${AL_ROOT}/simulation/iteration_${ITER}"
TRAIN_DIR="${AL_ROOT}/training/iteration_${ITER}"
mkdir -p "${SIM_DIR}" "${TRAIN_DIR}"
cd "${SIM_DIR}"

echo "=== UHAL AL iteration ${ITER} / ${N_ITERS} ==="
echo "Model: ${MODEL_NAME}   n_select: ${N_SELECT}   T: ${T_VAL}K"

# Locate the foundation model for this iteration.
if [[ "${ITER}" -eq 1 ]]; then
    MODEL_SRC="${FILE_ROOT}/${MODEL_NAME}.model"
else
    MODEL_SRC="${AL_ROOT}/training/iteration_$((ITER - 1))/${MODEL_NAME}.model"
fi
cp "${MODEL_SRC}" .

UHAL_ARGS=(
    --init_xyz "${INIT_XYZ}"
    --mace_model "${MODEL_NAME}.model"
    --fname "${RUN_NAME}"
    --device cuda
    --committee_size "${COMMITTEE_SIZE}"
    --prior_weight "${PRIOR_WEIGHT}"
    --energy_weight "${ENERGY_WEIGHT}"
    --forces_weight "${FORCES_WEIGHT}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --tau_rel "${TAU_REL}"
    --tau_hist "${TAU_HIST}"
    --selector_threshold "${SELECTOR_THRESHOLD}"
    --selector_delay "${SELECTOR_DELAY}"
    --selector_mixing "${SELECTOR_MIXING}"
    --selector_thresh_mul "${SELECTOR_THRESH_MUL}"
    --selector_interval "${SELECTOR_INTERVAL}"
    --T "${T_VAL}"
    --friction "${FRICTION}"
    --timestep_fs "${TIMESTEP_FS}"
    --n_silent_burnin "${N_SILENT_BURNIN}"
    --n_burnin_md "${N_BURNIN_MD}"
    --every_burnin "${EVERY_BURNIN}"
    --nsteps "${NSTEPS}"
    --n_run "${N_RUN}"
    --n_select "${N_SELECT}"
    --npz_path "${NPZ}"
)

LABEL_ARGS=(
    --fname "${RUN_NAME}"
    --oracle_model "${ORACLE_PATH}"
    --npz_path "${NPZ}"
)

if [[ "${ITER}" -ge 2 ]]; then
    cp "${AL_ROOT}/training/iteration_$((ITER - 1))/training_data_it_$((ITER - 1)).xyz" prev_samples.xyz
    LABEL_ARGS+=(--prev_samples prev_samples.xyz)
fi

# ── UHAL exploration MD ────────────────────────────────────────────────────
python "${SCRIPT_ROOT}/uhal_run/run_uhal_mace.py" "${UHAL_ARGS[@]}"

# ── Oracle-label the online-selected data (no FPS -- HAL score already selected) ──
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

NEW_MODEL="mace_off_32_uhal_it_${ITER}"

mace_run_train \
    --config           "${SCRIPT_ROOT}/training/lora_input.yaml" \
    --name             "${NEW_MODEL}_trained" \
    --train_file       "training_data_it_${ITER}.xyz" \
    --foundation_model "${MODEL_NAME}.model"

mv "${NEW_MODEL}_trained_stagetwo.model" "${NEW_MODEL}.model"
echo "Model: ${TRAIN_DIR}/${NEW_MODEL}.model"

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
        --export=ALL,ITER="${NEXT_ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${NEW_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",N_RUN="${N_RUN}",COMMITTEE_SIZE="${COMMITTEE_SIZE}",PRIOR_WEIGHT="${PRIOR_WEIGHT}",ENERGY_WEIGHT="${ENERGY_WEIGHT}",FORCES_WEIGHT="${FORCES_WEIGHT}",BATCH_SIZE="${BATCH_SIZE}",TAU_REL="${TAU_REL}",TAU_HIST="${TAU_HIST}",SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD}",SELECTOR_DELAY="${SELECTOR_DELAY}",SELECTOR_MIXING="${SELECTOR_MIXING}",SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL}",SELECTOR_INTERVAL="${SELECTOR_INTERVAL}",T_VAL="${T_VAL}",FRICTION="${FRICTION}",TIMESTEP_FS="${TIMESTEP_FS}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}" \
        "${SCRIPT_ROOT}/slurm/example/iteration_uhal.slurm"
    echo "Submitted iteration ${NEXT_ITER}."
else
    echo "All ${N_ITERS} iterations complete."
fi
