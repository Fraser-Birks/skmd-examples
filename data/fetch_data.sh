#!/usr/bin/env bash
# Download the large inputs listed in data/MANIFEST.md from Zenodo into
# data/alanine_dipeptide/ and data/silicon_surface/.
#
# Usage:
#   bash data/fetch_data.sh [all|alanine|silicon|results]
#
# 'results' fetches the paper's per-run metrics (to replot Figs. 3-4); not part of 'all'.

set -euo pipefail

# Zenodo record 10.5281/zenodo.24011550
ZENODO_RECORD="${ZENODO_RECORD:-24011550}"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
WHICH="all"
for arg in "$@"; do
    case "${arg}" in
        all|alanine|silicon|results) WHICH="${arg}" ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: ${arg}" >&2; exit 1 ;;
    esac
done

# Third-party models are fetched from their authors' releases.
MACE_OFF_URL="https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_small.model"
MACE_MPA_URL="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"

fetch() {   # fetch <Zenodo file name or URL> <local path>
    local name="$1" dest="$2" url="${BASE_URL}/$1?download=1"
    [[ "${name}" == http* ]] && url="${name}"
    if [[ -s "${dest}" ]]; then
        echo "  [skip] ${dest#"${DATA_DIR}/"} (exists)"
        return
    fi
    mkdir -p "$(dirname "${dest}")"
    echo "  [get]  ${name##*/} -> ${dest#"${DATA_DIR}/"}"
    curl -fL --retry 3 -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
}

if [[ "${WHICH}" == "all" || "${WHICH}" == "alanine" ]]; then
    echo "Alanine dipeptide:"
    A="${DATA_DIR}/alanine_dipeptide"
    fetch "${MACE_OFF_URL}" "${A}/MACE-OFF23_small.model"
    for f in mace_off_32_0.1_percent.model training_set_0.1_percent.xyz \
             phi_psi_relaxed_pes_off23_small.npz test_set.xyz; do
        fetch "${f}" "${A}/${f}"
    done
fi

if [[ "${WHICH}" == "all" || "${WHICH}" == "silicon" ]]; then
    echo "Si(111) surface:"
    S="${DATA_DIR}/silicon_surface"
    fetch "${MACE_MPA_URL}" "${S}/mace-mpa-0-medium.model"
    # DFT OUTCARs are deposited as one archive containing
    # DFT_reference/{ref,pandey}/OUTCAR and DFT_relaxed/{ref,pandey}/OUTCAR.
    if [[ ! -s "${S}/DFT_reference/ref/OUTCAR" ]]; then
        fetch silicon_dft_outcars.tar.gz "${S}/silicon_dft_outcars.tar.gz"
        tar -xzf "${S}/silicon_dft_outcars.tar.gz" -C "${S}"
        rm "${S}/silicon_dft_outcars.tar.gz"
    else
        echo "  [skip] silicon_surface/DFT_reference (exists)"
    fi
fi

if [[ "${WHICH}" == "results" ]]; then
    echo "Paper results:"
    fetch skmd_results.tar.gz "${DATA_DIR}/skmd_results.tar.gz"
    tar -xzf "${DATA_DIR}/skmd_results.tar.gz" -C "${DATA_DIR}"
    rm "${DATA_DIR}/skmd_results.tar.gz"
fi

echo "Done. See data/MANIFEST.md for what each file is."
