#!/usr/bin/env bash
# Run N_SEEDS seeded unbiased Langevin (700 K, damping 0.1 ps, --a 0) AL loops via run_al_loop.sh,
# one --al-subdir per seed. NUM_ITERS/--n-select must match the other arms.
#
# Usage:
#   ./run_langevin_700k_seeds.sh NUM_ITERS N_SEEDS [--n-select N] [--seed-start S] [--venv PATH]
#
# Example (paper setting):
#   ./run_langevin_700k_seeds.sh 10 5 --n-select 100
#
# Output: data/alanine_dipeptide/langevin_700k_seed_<N>/{simulation,training}/iteration_*/

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

NUM_ITERS=""
N_SEEDS=""
N_SELECT=100
SEED_START=1
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)
T_VAL=700.0
DAMPING=0.1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-select)   N_SELECT="${2:?--n-select requires an integer}"; shift 2 ;;
        --seed-start) SEED_START="${2:?--seed-start requires an integer}"; shift 2 ;;
        --venv)       VENV_PATH="${2:?--venv requires a path}"; shift 2 ;;
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

echo "700K Langevin baseline seeds (full AL loop): ${NUM_ITERS} iterations x ${N_SEEDS} seeds, n-select=${N_SELECT}"
echo "NUM_ITERS/--n-select must match the other arms for a matched-budget comparison."

for ((i = 0; i < N_SEEDS; i++)); do
    SEED=$((SEED_START + i))
    RUN_NAME="langevin_700k_seed_${SEED}"
    AL_SUBDIR="langevin_700k_seed_${SEED}"
    echo "── seed ${SEED} (${AL_SUBDIR}) ──"
    "${SCRIPT_ROOT}/run_al_loop.sh" "${NUM_ITERS}" \
        --n-select "${N_SELECT}" \
        --venv "${VENV_PATH}" \
        --dyn-type underdamped \
        --T "${T_VAL}" \
        --damping "${DAMPING}" \
        --a 0 \
        --seed "${SEED}" \
        --run-name "${RUN_NAME}" \
        --al-subdir "${AL_SUBDIR}"
done
