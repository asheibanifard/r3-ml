#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METRICS_SCRIPT="${METRICS_SCRIPT:-${SCRIPT_DIR}/volume_metrics.py}"
PREDICTION="${PREDICTION:-${SCRIPT_DIR}/rec_block_z000_y001_x006.npy}"
REFERENCE_DATASET="${REFERENCE_DATASET:-raw}"
PREDICTION_DATASET="${PREDICTION_DATASET:-}"

# Apply the ground-truth min/max transform to both volumes.
# Use NORMALIZE=none only when both are already in exactly the same scale.
NORMALIZE="${NORMALIZE:-reference}"

# Search roots can be overridden, e.g.
# SEARCH_ROOT=/root/project ./run_volume_metrics.sh
SEARCH_ROOT="${SEARCH_ROOT:-/root/project}"

# Preferred expected filename.
REFERENCE_NAME="${REFERENCE_NAME:-block_z000_y001_x006.h5}"

OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/metrics.json}"
OUTPUT_CSV="${OUTPUT_CSV:-${SCRIPT_DIR}/metrics.csv}"

if [[ ! -f "${METRICS_SCRIPT}" ]]; then
    echo "Error: metrics script not found: ${METRICS_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${PREDICTION}" ]]; then
    echo "Error: prediction file not found: ${PREDICTION}" >&2
    exit 1
fi

if [[ "${PREDICTION##*.}" == "raw" && ! -f "${PREDICTION}.json" ]]; then
    echo "Error: RAW metadata file not found: ${PREDICTION}.json" >&2
    exit 1
fi

# A manually supplied absolute reference path takes priority.
if [[ -n "${REFERENCE:-}" ]]; then
    if [[ ! -f "${REFERENCE}" ]]; then
        echo "Error: supplied REFERENCE does not exist: ${REFERENCE}" >&2
        exit 1
    fi
    REFERENCE_PATH="${REFERENCE}"
else
    echo "Searching for ${REFERENCE_NAME} under ${SEARCH_ROOT} ..."

    mapfile -t MATCHES < <(
        find "${SEARCH_ROOT}" -type f \
            \( -name "${REFERENCE_NAME}" \
               -o -name "block_z0_y1_x6.h5" \
               -o -name "block_z000_y001_x006.hdf5" \) \
            2>/dev/null
    )

    if [[ ${#MATCHES[@]} -eq 0 ]]; then
        echo "Exact filename not found. Searching for similar HDF5 block files ..." >&2

        mapfile -t MATCHES < <(
            find "${SEARCH_ROOT}" -type f \
                \( -name "*.h5" -o -name "*.hdf5" \) \
                2>/dev/null |
            grep -E 'block.*z0*0.*y0*1.*x0*6|block.*z000.*y001.*x006' || true
        )
    fi

    if [[ ${#MATCHES[@]} -eq 0 ]]; then
        echo "Error: no matching reference HDF5 file was found." >&2
        echo "Try one of these:" >&2
        echo "  REFERENCE=/absolute/path/to/file.h5 ./run_volume_metrics.sh" >&2
        echo "  SEARCH_ROOT=/different/root ./run_volume_metrics.sh" >&2
        exit 1
    fi

    if [[ ${#MATCHES[@]} -gt 1 ]]; then
        echo "Multiple possible reference files found:" >&2
        printf '  %s\n' "${MATCHES[@]}" >&2
        echo "Set REFERENCE explicitly to choose one." >&2
        exit 1
    fi

    REFERENCE_PATH="${MATCHES[0]}"
fi

echo
echo "Reference : ${REFERENCE_PATH}"
echo "Prediction: ${PREDICTION}"
echo

CMD=(
    python "${METRICS_SCRIPT}"
    --reference "${REFERENCE_PATH}"
    --prediction "${PREDICTION}"
    --reference-dataset "${REFERENCE_DATASET}"
    --normalize "${NORMALIZE}"
    --output-json "${OUTPUT_JSON}"
    --output-csv "${OUTPUT_CSV}"
)

if [[ -n "${PREDICTION_DATASET}" ]]; then
    CMD+=(--prediction-dataset "${PREDICTION_DATASET}")
fi

"${CMD[@]}"

echo
echo "Saved JSON: ${OUTPUT_JSON}"
echo "Saved CSV : ${OUTPUT_CSV}"
