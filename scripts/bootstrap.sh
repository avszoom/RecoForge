#!/usr/bin/env bash
# Run the full RecoForge pipeline from a clean state.
#
# Used by the Streamlit app at first launch (when artifacts/ is empty)
# and as a CLI for local development ("redo everything from scratch").
#
# Total runtime: ~1 minute on a modern laptop / Streamlit Cloud free tier.
# Idempotent: if some artifacts already exist they get overwritten.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python"   # fall back to system python (Streamlit Cloud)

# OMP env vars to sidestep the macOS / Linux libomp interactions when
# torch + faiss share a process. Harmless on platforms that don't need them.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_INIT_AT_FORK=FALSE

steps=(
  "src.data.generate_dataset"
  "src.models.text_features"
  "src.models.train_two_tower"
  "src.models.export_embeddings"
  "src.indexing.build_faiss"
)

for mod in "${steps[@]}"; do
  echo "▶ $mod"
  "$PY" -m "$mod"
  echo
done

echo "✓ Bootstrap complete."
