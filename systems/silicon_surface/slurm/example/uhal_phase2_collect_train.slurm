#!/bin/bash
#SBATCH --job-name=si_al_uhal_p2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:45:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# LoRA fine-tune + export + eval: a few minutes on one GPU.

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"
: "${FILE_ROOT:?FILE_ROOT not set}"
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-30}"
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
N_SILENT_BURNIN="${N_SILENT_BURNIN:-4000}"
N_BURNIN_MD="${N_BURNIN_MD:-6000}"
EVERY_BURNIN="${EVERY_BURNIN:-200}"
: "${N_FROZEN:?N_FROZEN not set}"
GPU_DFT="${GPU_DFT:-1}"

RUN_NAME="${RUN_NAME:-uhal}"
PREV_ITER=$((ITER - 1))
TRAIN_DIR="${AL_ROOT}/training/iteration_${ITER}"
NEW_MODEL="si_slab_uhal_lora_it_${ITER}"

mkdir -p "${TRAIN_DIR}"

echo "=== UHAL Phase 2: iteration ${ITER} / ${N_ITERS} ==="

# ── 1. Collect DFT results -> cumulative XYZ ─────────────────────────────────
if [[ "${ITER}" -eq 1 ]]; then
    PREV_XYZ_ARG=""
else
    PREV_XYZ="${AL_ROOT}/training/iteration_${PREV_ITER}/training_data_it_${PREV_ITER}.xyz"
    PREV_XYZ_ARG="--prev-xyz ${PREV_XYZ}"
fi

python "${SCRIPT_ROOT}/dft/collect_dft_results.py" \
    --dft-inputs "${AL_ROOT}/simulation/iteration_${ITER}/${RUN_NAME}/dft_inputs" \
    ${PREV_XYZ_ARG} \
    --output     "${TRAIN_DIR}/training_data_it_${ITER}.xyz" \
    --iteration  "${ITER}"

# ── 2. LoRA fine-tuning ────────────────────────────────────────────────────────
cd "${TRAIN_DIR}"

if [[ "${ITER}" -eq 1 ]]; then
    cp "${FILE_ROOT}/${MODEL_NAME}.model" .
else
    cp "${AL_ROOT}/training/iteration_${PREV_ITER}/${MODEL_NAME}.model" .
fi

mace_run_train \
    --config           "${SCRIPT_ROOT}/training/lora_input.yaml" \
    --name             "${NEW_MODEL}_trained" \
    --train_file       "training_data_it_${ITER}.xyz" \
    --foundation_model "${MODEL_NAME}.model"

mv "${NEW_MODEL}_trained_stagetwo.model" "${NEW_MODEL}.model"
echo "Model: ${TRAIN_DIR}/${NEW_MODEL}.model"

# ── 3. Export JSON (shared analysis pipeline needs LAMMPS/symmetrix) ──────────
symmetrix_extract_mace \
    -m "${NEW_MODEL}.model" \
    --atomic-numbers 14
echo "JSON:  ${TRAIN_DIR}/${NEW_MODEL}-14.json"

# ── 4. Evaluate on training set (and a held-out test set if one exists) ───────
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
    echo "[uhal_phase2] no ${FILE_ROOT}/test_set.xyz found; skipping held-out evaluation"
fi

# ── 5. Propagate to next iteration ─────────────────────────────────────────────
if [[ "${ITER}" -lt "${N_ITERS}" ]]; then
    NEXT_ITER=$((ITER + 1))
    mkdir -p "${AL_ROOT}/simulation/iteration_${NEXT_ITER}"
    mkdir -p "${AL_ROOT}/training/iteration_${NEXT_ITER}"

    sbatch \
        --export=ALL,ITER="${NEXT_ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${NEW_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",NSTEPS="${NSTEPS}",N_RUN="${N_RUN}",N_FROZEN="${N_FROZEN}",COMMITTEE_SIZE="${COMMITTEE_SIZE}",PRIOR_WEIGHT="${PRIOR_WEIGHT}",ENERGY_WEIGHT="${ENERGY_WEIGHT}",FORCES_WEIGHT="${FORCES_WEIGHT}",BATCH_SIZE="${BATCH_SIZE}",SEED="${SEED}",TAU_REL="${TAU_REL}",TAU_HIST="${TAU_HIST}",SELECTOR_THRESHOLD="${SELECTOR_THRESHOLD}",SELECTOR_DELAY="${SELECTOR_DELAY}",SELECTOR_MIXING="${SELECTOR_MIXING}",SELECTOR_THRESH_MUL="${SELECTOR_THRESH_MUL}",SELECTOR_INTERVAL="${SELECTOR_INTERVAL}",T_VAL="${T_VAL}",FRICTION="${FRICTION}",TIMESTEP_FS="${TIMESTEP_FS}",N_SILENT_BURNIN="${N_SILENT_BURNIN}",N_BURNIN_MD="${N_BURNIN_MD}",EVERY_BURNIN="${EVERY_BURNIN}",GPU_DFT="${GPU_DFT}" \
        "${SCRIPT_ROOT}/slurm/example/uhal_phase1_md_sample.slurm"
    echo "Submitted UHAL Phase 1 for iteration ${NEXT_ITER}."
else
    echo "All ${N_ITERS} iterations complete."
fi
