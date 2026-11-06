#!/usr/bin/env bash
# Run N_SEEDS seeded a-SKMD (online acquisition, Eq. 8) AL loops, one --al-subdir per seed; match NUM_ITERS/--n-select across arms.
# Usage: ./run_askmd_seeds.sh NUM_ITERS N_SEEDS [--n-select N] [--seed-start S] [--venv PATH]

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

NUM_ITERS=""
N_SEEDS=""
N_SELECT=30
SEED_START=1
VENV_PATH="${SKMD_VENV:-}"   # venv with skmd + LAMMPS/symmetrix/MACE (or pass --venv)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-select)   N_SELECT="${2:?--n-select requires an integer}"; shift 2 ;;
        --seed-start) SEED_START="${2:?--seed-start requires an integer}"; shift 2 ;;
        --venv)       VENV_PATH="${2:?--venv requires a path}"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1 (any run_al_loop_askmd.sh flag not listed above isn't forwarded — edit this script if you need one)" >&2
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

echo "a-SKMD seeds (full AL loop): ${NUM_ITERS} iterations x ${N_SEEDS} seeds, n-select=${N_SELECT}"
echo "NUM_ITERS/--n-select must match the other arms for a matched-budget comparison."

for ((i = 0; i < N_SEEDS; i++)); do
    SEED=$((SEED_START + i))
    RUN_NAME="askmd_seed_${SEED}"
    AL_SUBDIR="askmd_seed_${SEED}"
    echo "── seed ${SEED} (${AL_SUBDIR}) ──"
    "${SCRIPT_ROOT}/run_al_loop_askmd.sh" "${NUM_ITERS}" \
        --n-select "${N_SELECT}" \
        --venv "${VENV_PATH}" \
        --seed "${SEED}" \
        --run-name "${RUN_NAME}" \
        --al-subdir "${AL_SUBDIR}"
done
