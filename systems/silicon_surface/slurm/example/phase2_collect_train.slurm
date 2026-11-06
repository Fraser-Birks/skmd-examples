#!/bin/bash
#SBATCH --job-name=si_al_p2
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
# SKMD dynamics: see run_al_loop_skmd_offline.sh and run_skmd_mace.py --b_mode help.
NSTEPS="${NSTEPS:-200}"
RELAX_START="${RELAX_START:-160}"
N_RUN="${N_RUN:-500}"
A_VAL="${A_VAL:-4.0}"
B_VAL="${B_VAL:-1.0}"
B_MODE="${B_MODE:-adaptive-median}"
B_MAX_MULT="${B_MAX_MULT:-15.0}"
T_VAL="${T_VAL:-50.0}"
DAMPING="${DAMPING:-75.0}"
DYN_TYPE="${DYN_TYPE:-overdamped}"
THRESH="${THRESH:-}"
BUFFER="${BUFFER:-}"
B_STABILITY_TOL="${B_STABILITY_TOL:-}"
B_STABILITY_WINDOW="${B_STABILITY_WINDOW:-}"
: "${N_FROZEN:?N_FROZEN not set}"
GPU_DFT="${GPU_DFT:-1}"

RUN_NAME="${RUN_NAME:-skmd_run}"
SEED="${SEED:-123456}"
PREV_ITER=$((ITER - 1))
TRAIN_DIR="${AL_ROOT}/training/iteration_${ITER}"
NEW_MODEL="si_slab_skmd_lora_it_${ITER}"

mkdir -p "${TRAIN_DIR}"

echo "=== Phase 2: iteration ${ITER} / ${N_ITERS} ==="

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

# ── 3. Export JSON for LAMMPS ──────────────────────────────────────────────────
symmetrix_extract_mace \
    -m "${NEW_MODEL}.model" \
    --atomic-numbers 14
echo "JSON:  ${TRAIN_DIR}/${NEW_MODEL}-14.json"

# ── 4. Evaluate on training set (and a held-out test set if one exists) ───────
# No test_set.xyz ships (paper metric: reconstruction energy); add FILE_ROOT/test_set.xyz for held-out metrics.
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
    echo "[phase2] no ${FILE_ROOT}/test_set.xyz found; skipping held-out evaluation"
fi

# ── 5. Propagate to next iteration ─────────────────────────────────────────────
if [[ "${ITER}" -lt "${N_ITERS}" ]]; then
    NEXT_ITER=$((ITER + 1))
    mkdir -p "${AL_ROOT}/simulation/iteration_${NEXT_ITER}"
    mkdir -p "${AL_ROOT}/training/iteration_${NEXT_ITER}"

    sbatch \
        --export=ALL,ITER="${NEXT_ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${NEW_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",N_FROZEN="${N_FROZEN}",DYN_TYPE="${DYN_TYPE}",THRESH="${THRESH}",BUFFER="${BUFFER}",B_STABILITY_TOL="${B_STABILITY_TOL}",B_STABILITY_WINDOW="${B_STABILITY_WINDOW}",GPU_DFT="${GPU_DFT}" \
        "${SCRIPT_ROOT}/slurm/example/phase1_md_sample.slurm"
    echo "Submitted Phase 1 for iteration ${NEXT_ITER}."
else
    echo "All ${N_ITERS} iterations complete."
fi
