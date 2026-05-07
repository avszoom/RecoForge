#!/usr/bin/env bash
# Run each test file in its own pytest invocation.
#
# Why: faiss-cpu and PyTorch both ship their own libomp. On macOS Apple
# Silicon, having both loaded into a single long-lived process can lead to
# intermittent segfaults inside faiss.search after PyTorch has loaded its
# state_dict. Running tests file-by-file in fresh processes sidesteps this
# entirely — each pytest run only loads the deps that file actually needs.
#
# Usage:
#   scripts/test.sh            # run every test file in tests/
#   scripts/test.sh -k pattern # forwarded to each pytest invocation
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
EXTRA_ARGS=("$@")

# Discover test files in stable order. Glob expands sorted by default on macOS.
FILES=()
for f in tests/test_*.py; do
  [ -e "$f" ] || continue
  FILES+=("$f")
done

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no test files found in tests/"
  exit 1
fi

echo "RecoForge test runner — ${#FILES[@]} file(s)"
echo

failed=""
for f in "${FILES[@]}"; do
  echo "=== $f ==="
  if "$PY" -m pytest "$f" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
    echo
  else
    echo "FAILED: $f"
    failed="$failed $f"
    echo
  fi
done

echo "──────────────────────────────────────────────"
if [ -z "$failed" ]; then
  echo "All ${#FILES[@]} test files passed."
  exit 0
else
  echo "FAILED:"
  for f in $failed; do echo "  - $f"; done
  exit 1
fi
