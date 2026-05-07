#!/usr/bin/env bash
# Launch the Streamlit demo with the env-var workarounds for the
# macOS Apple Silicon faiss-cpu + torch libomp combo.
#
# Symptom these vars prevent:
#     OMP: Error #179: Function pthread_mutex_init failed:
#     OMP: System error #22: Invalid argument
#     zsh: segmentation fault  .venv/bin/streamlit run app/streamlit_app.py
#
# The vars MUST be exported before python starts — setting them from inside
# Python is too late because Streamlit's transitive imports may have already
# loaded one of the libomp instances.
#
# OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 — single-threaded math, sidesteps the
# OMP-init race entirely. Performance hit at our 1k-user / 4k-item POC scale
# is unmeasurable.
#
# Usage:
#   ./scripts/run_app.sh
#   ./scripts/run_app.sh --server.port 8765    # extra args forwarded
set -euo pipefail
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_INIT_AT_FORK=FALSE

exec "${PYTHON:-.venv/bin/python}" -m streamlit run app/streamlit_app.py ${@+"$@"}
