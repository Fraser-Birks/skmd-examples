#!/bin/bash
#SBATCH --job-name=si_al_p1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:00:00

# EXAMPLE for Quadro RTX 6000 GPUs (24 GB), 4 cores/job: edit #SBATCH lines for your site;
# set SKMD_MODULES to your builds' modules (README, "Cluster environment").

# GPU-resident SKMD exploration (run_skmd_mace.py --kokkos), ~0.56 h on a Quadro RTX 6000, plus FPS selection.

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -n "${SKMD_MODULES:-}" ]]; then module purge; module load ${SKMD_MODULES}; fi

source "${VENV_PATH:?VENV_PATH not set}/bin/activate"

set -euo pipefail

: "${AL_ROOT:?AL_ROOT not set}"          # .../silicon_surface/skmd
: "${FILE_ROOT:?FILE_ROOT not set}"      # .../silicon_surface
: "${SCRIPT_ROOT:?SCRIPT_ROOT not set}"  # .../systems/silicon_surface
: "${ITER:?ITER not set}"
: "${N_ITERS:?N_ITERS not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
N_SELECT="${N_SELECT:-30}"

RUN_NAME="${RUN_NAME:-skmd_run}"
SEED="${SEED:-123456}"
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
N_FROZEN="${N_FROZEN:?N_FROZEN not set — see prepare_lammps_data.py}"
# Optional adaptive stopping (run_skmd_mace.py --thresh/--buffer); empty = off.
THRESH="${THRESH:-}"
BUFFER="${BUFFER:-}"
# Optional early exit once adaptive b stabilizes (--b_stability_tol/--b_stability_window).
B_STABILITY_TOL="${B_STABILITY_TOL:-}"
B_STABILITY_WINDOW="${B_STABILITY_WINDOW:-}"
# GPU_DFT applies to every iteration, so it is forwarded to Phase 2 and never unset.
GPU_DFT="${GPU_DFT:-1}"

INIT_POS_BASENAME="configs_before_it_$((NSTEPS - 1)).xyz"

LAMMPS_DATA="${FILE_ROOT}/silicon_slab.data"
INIT_XYZ="${FILE_ROOT}/relaxed_struct_non_pandey.xyz"

SIM_DIR="${AL_ROOT}/simulation/iteration_${ITER}"
mkdir -p "${SIM_DIR}"
cd "${SIM_DIR}"

echo "=== Phase 1: iteration ${ITER} / ${N_ITERS} ==="
echo "Model: ${MODEL_NAME}   Selections: ${N_SELECT}"

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
    -n "${N_SELECT}"
)

if [[ "${ITER}" -ge 2 ]]; then
    PREV_SIM_DIR="${AL_ROOT}/simulation/iteration_$((ITER - 1))/${RUN_NAME}"
    cp "${PREV_SIM_DIR}/traj_data/${INIT_POS_BASENAME}" .
    RUN_ARGS+=(--initial_particle_positions "${INIT_POS_BASENAME}")
    POST_ARGS+=(--initial_particle_positions "${INIT_POS_BASENAME}")

    # Iteration 1 wrote only diverse_subset.xyz (no --prev_samples yet).
    if [[ "${ITER}" -eq 2 ]]; then
        cp "${PREV_SIM_DIR}/diverse_subset.xyz" prev_samples.xyz
    else
        cp "${PREV_SIM_DIR}/new_and_previous_samples.xyz" prev_samples.xyz
    fi
    DRAW_ARGS+=(--prev_samples prev_samples.xyz)
fi

# ── SKMD exploration MD ───────────────────────────────────────────────────────
# LAMMPS/symmetrix may crash harmlessly at interpreter teardown (exit 134/139) after finishing;
# treat the final snapshot as the completion marker instead of the exit code.
COMPLETION_MARKER="${SIM_DIR}/${RUN_NAME}/traj_data/${INIT_POS_BASENAME}"
if [[ -n "${SKIP_MD:-}" ]]; then
    # SKIP_MD=1 (via --export): MD already completed, jump to post-processing.
    [[ -f "${COMPLETION_MARKER}" ]] || {
        echo "SKIP_MD set but ${COMPLETION_MARKER} not found — nothing to skip to, aborting" >&2
        exit 1
    }
    echo "SKIP_MD set and ${COMPLETION_MARKER} exists — skipping run_skmd_mace.py"
    # --export=ALL downstream: unset so SKIP_MD does not leak into later jobs.
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

# ── Post-process trajectory + FPS diversity selection ─────────────────────────
python "${SCRIPT_ROOT}/skmd_run/post_process_traj.py" "${POST_ARGS[@]}"
python "${SCRIPT_ROOT}/skmd_run/draw_samples.py" "${DRAW_ARGS[@]}"

# ── Write VASP inputs for the newly selected structures ───────────────────────
# config_cpu/gpu.yaml ncore/kpar match dft_array/gpu_dft_array MPI layouts.
if [[ -n "${GPU_DFT}" ]]; then
    DFT_CONFIG="${SCRIPT_ROOT}/dft/config_gpu.yaml"
    DFT_ARRAY_SLURM="${SCRIPT_ROOT}/slurm/example/gpu_dft_array.slurm"
else
    DFT_CONFIG="${SCRIPT_ROOT}/dft/config_cpu.yaml"
    DFT_ARRAY_SLURM="${SCRIPT_ROOT}/slurm/example/dft_array.slurm"
fi
python "${SCRIPT_ROOT}/dft/prepare_dft_inputs.py" \
    --subset "${RUN_NAME}/diverse_subset.xyz" \
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
    --export=ALL,ITER="${ITER}",N_ITERS="${N_ITERS}",MODEL_NAME="${MODEL_NAME}",N_SELECT="${N_SELECT}",AL_ROOT="${AL_ROOT}",FILE_ROOT="${FILE_ROOT}",SCRIPT_ROOT="${SCRIPT_ROOT}",VENV_PATH="${VENV_PATH}",RUN_NAME="${RUN_NAME}",SEED="${SEED}",NSTEPS="${NSTEPS}",RELAX_START="${RELAX_START}",N_RUN="${N_RUN}",A_VAL="${A_VAL}",B_VAL="${B_VAL}",B_MODE="${B_MODE}",B_MAX_MULT="${B_MAX_MULT}",T_VAL="${T_VAL}",DAMPING="${DAMPING}",N_FROZEN="${N_FROZEN}",DYN_TYPE="${DYN_TYPE}",THRESH="${THRESH}",BUFFER="${BUFFER}",B_STABILITY_TOL="${B_STABILITY_TOL}",B_STABILITY_WINDOW="${B_STABILITY_WINDOW}",GPU_DFT="${GPU_DFT}" \
    "${SCRIPT_ROOT}/slurm/example/phase2_collect_train.slurm"

echo "Phase 1 complete.  DFT job ${DFT_JOB} queued; Phase 2 queued with dependency."
