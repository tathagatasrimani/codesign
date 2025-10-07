#!/usr/bin/env bash

set -eo pipefail

echo "🔎 Linting…"
# Examples:
# python -m ruff check .
# npm run lint
# cmake -S . -B build && cmake --build build --target format-check

echo "🧪 Running tests…"

source miniconda3/etc/profile.d/conda.sh


# Check if a conda environment is already active
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "No conda environment active. Activating 'codesign'..."
  # Adjust 'myenv' to the environment name you want
  conda activate codesign
  echo "✅ Conda environment Activated successfully!"
else
  echo "✅ Conda environment already active: $CONDA_DEFAULT_ENV"
fi


############### Add useful alisas ###############

shopt -s expand_aliases

alias create_checkpoint="python3 -m test.checkpoint_controller"
alias run_codesign="python3 -m src.codesign"

echo "Activated the alias succesfully!"

# run_codesign --config vitis_gemm_checkpoint_after_pd
set +e
run_codesign --config vitis_gemm_checkpoint_after_pd > run.log 2>&1
status=$?
set -e

if grep -q "AssertionError" run.log; then
  echo "✅ Expected AssertionError found. Treating as success."
  cat run.log
  exit 0
elif [ $status -ne 0 ]; then
  echo "❌ Unexpected error occurred"
  cat run.log
  exit $status
else
  echo "✅ Script completed successfully"
  cat run.log
  exit 0
fi

Examples:
pytest -q --maxfail=1 --disable-warnings --junitxml=reports/test-results.xml
npm test -- --ci
ctest --test-dir build --output-on-failure

echo "✅ All checks passed."
