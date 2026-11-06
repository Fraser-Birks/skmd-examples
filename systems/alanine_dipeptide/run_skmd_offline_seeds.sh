#!/usr/bin/env bash
# Run N_SEEDS seeded offline SKMD (FPS selection) AL loops via run_al_loop.sh, one --al-subdir per seed;
# a/b held constant after b freezes (paper App. E.2). NUM_ITERS/--n-select must match the other arms.
#
# Usage:
#   ./run_skmd_offline_seeds.sh NUM_ITERS N_SEEDS [--n-select N] [--seed-start S] [--venv PATH] [--dyn-type TYPE] [--a VAL]
#
# Example (paper setting):
#   ./run_skmd_offline_seeds.sh 10 5 --n-select 100
#
# Output: data/alanine_dipeptide/skmd_offline_seed_<N>/{simulation,training}/iteration_*/

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

NUM_ITERS=""
N_SEEDS=""
N_SELECT=100
SEED_START=1
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)
DYN_TYPE=overdamped
A_VAL=3.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-select)   N_SELECT="${2:?--n-select requires an integer}"; shift 2 ;;
        --seed-start) SEED_START="${2:?--seed-start requires an integer}"; shift 2 ;;
        --venv)       VENV_PATH="${2:?--venv requires a path}"; shift 2 ;;
        --dyn-type)   DYN_TYPE="${2:?--dyn-type requires overdamped or underdamped}"; shift 2 ;;
        --a)          A_VAL="${2:?--a requires a number}"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1 (any run_al_loop.sh flag not listed above isn't forwarded — edit this script if you need one)" >&2
            exit 1 ;;
        *)
            if [[ -z "${NUM_ITERS}" ]]; then
                NUM_ITERS="$1"
            elif [[ -z "${N_SEEDS}" ]]; then
                N_SEEDS="$1"
            else
                echo "Unexpected argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

: "${NUM_ITERS:?Usage: $0 NUM_ITERS N_SEEDS [--n-select N] [--seed-start S]}"
: "${N_SEEDS:?Usage: $0 NUM_ITERS N_SEEDS [--n-select N] [--seed-start S]}"
[[ "${NUM_ITERS}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ITERS must be a positive integer" >&2; exit 1; }
[[ "${N_SEEDS}" =~ ^[1-9][0-9]*$ ]] || { echo "N_SEEDS must be a positive integer" >&2; exit 1; }
: "${VENV_PATH:?set SKMD_VENV=/path/to/venv (or pass --venv PATH)}"

echo "SKMD-offline seeds (full AL loop): ${NUM_ITERS} iterations x ${N_SEEDS} seeds, n-select=${N_SELECT}, a=${A_VAL}"
echo "NUM_ITERS/--n-select must match the other arms for a matched-budget comparison."

for ((i = 0; i < N_SEEDS; i++)); do
    SEED=$((SEED_START + i))
    RUN_NAME="overdamped_a_${A_VAL}_T_50_seed_${SEED}"
    AL_SUBDIR="skmd_offline_seed_${SEED}"
    echo "── seed ${SEED} (${AL_SUBDIR}) ──"
    "${SCRIPT_ROOT}/run_al_loop.sh" "${NUM_ITERS}" \
        --n-select "${N_SELECT}" \
        --venv "${VENV_PATH}" \
        --dyn-type "${DYN_TYPE}" \
        --a "${A_VAL}" \
        --seed "${SEED}" \
        --run-name "${RUN_NAME}" \
        --al-subdir "${AL_SUBDIR}" \
        --a_relax_mult 1.0 \
        --b_relax_mult 1.0
done
