#!/bin/bash
#SBATCH --job-name=ad_al_it
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=01:30:00

# EXAMPLE SLURM script (Quadro RTX 6000 GPU, 4 cores): adapt the #SBATCH lines to your site and set
# SKMD_MODULES to the modules your builds need (README, "Cluster environment").

# One self-chaining GPU job per AL iteration: SKMD MD, FPS selection + MACE-oracle labelling,
# LoRA fine-tuning, eval.

# ── Environment ──────────────────────────────────────────────────────────────
# Optional: SKMD_MODULES = space-separated module list to load.
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"          # .../alanine_dipeptide/skmd
: "${FILE_ROOT:?FILE_ROOT not set}"      # .../alanine_dipeptide (data root)
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"  # .../systems/alanine_dipeptide
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-100}"

# Defaults; the run_*_seeds.sh wrappers override these per arm/seed.
RUN_NAME="${RUN_NAME:-overdamped_a_3_T_50}"
SEED="${SEED:-123456}"

NSTEPS="${NSTEPS:-300}"
RELAX_START="${RELAX_START:-250}"
N_RUN="${N_RUN:-500}"
A_VAL="${A_VAL:-3.0}"
B_VAL="${B_VAL:-0.7}"
B_MODE="${B_MODE:-adaptive}"
B_MAX_MULT="${B_MAX_MULT:-3.0}"
T_VAL="${T_VAL:-50.0}"
DAMPING="${DAMPING:-75.0}"
DYN_TYPE="${DYN_TYPE:-overdamped}"
THRESH="${THRESH:-}"
BUFFER="${BUFFER:-}"
B_STABILITY_TOL="${B_STABILITY_TOL:-}"
B_STABILITY_WINDOW="${B_STABILITY_WINDOW:-}"
A_RELAX_MULT="${A_RELAX_MULT:-0.5}"
B_RELAX_MULT="${B_RELAX_MULT:-0.5}"

INIT_POS_BASENAME="configs_before_it_$((NSTEPS - 1)).xyz"

LAMMPS_DATA="${FILE_ROOT}/ace-ala-nme.data"
INIT_XYZ="${FILE_ROOT}/ace-ala-nme.xyz"
NPZ="${FILE_ROOT}/phi_psi_relaxed_pes_off23_small.npz"
ORACLE_PATH="${FILE_ROOT}/MACE-OFF23_small.model"
ATOMIC_NUMBERS=(1 6 7 8)

SIM_DIR="${AL_ROOT}/simulation/iteration_${ITER}"
TRAIN_DIR="${AL_ROOT}/training/iteration_${ITER}"
mkdir -p "${SIM_DIR}" "${TRAIN_DIR}"
cd "${SIM_DIR}"

echo "=== SKMD AL iteration ${ITER} / ${N_ITERS} ==="
echo "Model: ${MODEL_NAME}   Selections: ${N_SELECT}"

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
    --dyn_type "${DYN_TYPE}"
    --seed "${SEED}"
    --a_relax_mult "${A_RELAX_MULT}"
    --b_relax_mult "${B_RELAX_MULT}"
    --kokkos
    --kokkos_gpus 1
)

if [[ -n "${THRESH}" ]]; then
    RUN_ARGS+=(--thresh "${THRESH}")
fi
if [[ -n "${BUFFER}" ]]; then
    RUN_ARGS+=(--buffer "${BUFFER}")
fi
if [[ -n "${B_STABILITY_TOL}" ]]; then
    RUN_ARGS+=(--b_stability_tol "${B_STABILITY_TOL}")
fi
if [[ -n "${B_STABILITY_WINDOW}" ]]; then
    RUN_ARGS+=(--b_stability_window "${B_STABILITY_WINDOW}")
fi

POST_ARGS=(
    --fname "${RUN_NAME}"
    --nsteps "${NSTEPS}"
)

DRAW_ARGS=(
    --fname "${RUN_NAME}"
    --npz_path "${NPZ}"
    -n "${N_SELECT}"
    --oracle_model "${ORACLE_PATH}"
)

if [[ "${ITER}" -ge 2 ]]; then
    PREV_SIM_DIR="${AL_ROOT}/simulation/iteration_$((ITER - 1))/${RUN_NAME}"
    cp "${PREV_SIM_DIR}/traj_data/${INIT_POS_BASENAME}" .
    RUN_ARGS+=(--initial_particle_positions "${INIT_POS_BASENAME}")
    POST_ARGS+=(--initial_particle_positions "${INIT_POS_BASENAME}")

    cp "${AL_ROOT}/training/iteration_$((ITER - 1))/training_data_it_$((ITER - 1)).xyz" prev_samples.xyz
    DRAW_ARGS+=(--prev_samples prev_samples.xyz)
fi

# ── SKMD exploration MD ───────────────────────────────────────────────────────
# LAMMPS/symmetrix can crash harmlessly on interpreter teardown (exit 139/134) after finishing,
# so success is judged by the final snapshot file, which iteration N+1 needs anyway.
COMPLETION_MARKER="${SIM_DIR}/${RUN_NAME}/traj_data/${INIT_POS_BASENAME}"
if [[ -n "${SKIP_MD:-}" ]]; then
    # Manual resubmit: MD already completed, jump straight to post-processing.
    [[ -f "${COMPLETION_MARKER}" ]] || {
        echo "SKIP_MD set but ${COMPLETION_MARKER} not found — nothing to skip to, aborting" >&2
        exit 1
    }
    echo "SKIP_MD set and ${COMPLETION_MARKER} exists — skipping run_skmd_mace.py"
    # Don't leak SKIP_MD into the next iteration via --export=ALL.
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

# ── Post-process trajectory + FPS diversity selection + oracle labelling ──────
python "${SCRIPT_ROOT}/skmd_run/post_process_traj.py" "${POST_ARGS[@]}"
python "${SCRIPT_ROOT}/skmd_run/draw_samples.py" "${DRAW_ARGS[@]}"

# ── Assemble this iteration's training set ─────────────────────────────────────
cd "${TRAIN_DIR}"
if [[ "${ITER}" -eq 1 ]]; then
    cp "${SIM_DIR}/${RUN_NAME}/diverse_subset.xyz" "training_data_it_${ITER}.xyz"
else
    cp "${SIM_DIR}/${RUN_NAME}/new_and_previous_samples.xyz" "training_data_it_${ITER}.xyz"
fi

# ── LoRA fine-tuning ────────────────────────────────────────────────────────────
cp "${SIM_DIR}/${MODEL_NAME}.model" .

NEW_MODEL="mace_off_32_skmd_it_${ITER}"

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
        --export=ALL,ITER="${NEXT_ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${NEW_MODEL}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",DYN_TYPE="${DYN_TYPE}",THRESH="${THRESH}",BUFFER="${BUFFER}",B_STABILITY_TOL="${B_STABILITY_TOL}",B_STABILITY_WINDOW="${B_STABILITY_WINDOW}",A_RELAX_MULT="${A_RELAX_MULT}",B_RELAX_MULT="${B_RELAX_MULT}" \
        "${SCRIPT_ROOT}/slurm/example/iteration.slurm"
    echo "Submitted iteration ${NEXT_ITER}."
else
    echo "All ${N_ITERS} iterations complete."
fi
