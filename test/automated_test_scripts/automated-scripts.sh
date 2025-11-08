#!/usr/bin/env bash

set -eo pipefail

shopt -s expand_aliases

# echo "🔎 Linting…"
# Examples:
# python -m ruff check .
# npm run lint
# cmake -S . -B build && cmake --build build --target format-check

# OPENROAD_PRE_INSTALLED_BIN_PATH="../../../../deps/OpenROAD/build/bin/openroad"

# echo "🧪 Running tests…"

# source miniconda3/etc/profile.d/conda.sh

OPENROAD_PRE_INSTALLED=1
source full_env_start.sh

# # Check if a conda environment is already active
# if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
#   echo "No conda environment active. Activating 'codesign'..."
#   # Adjust 'myenv' to the environment name you want
#   conda activate codesign
#   echo "✅ Conda environment Activated successfully!"
# else
#   echo "✅ Conda environment already active: $CONDA_DEFAULT_ENV"
# fi


############### Add useful alisas ###############



# alias create_checkpoint="python3 -m test.checkpoint_controller"
# alias run_codesign="python3 -m src.codesign"
# alias run_regression="python3 -m test.regression_run"

# echo "Activated the alias succesfully!"

# Run regression and propagate its exit code directly.
# Use 'set +e' so the script can capture the exit code instead of exiting immediately.
set +e
run_regression -l auto_tests/auto-testlist.yaml -m 10 --preinstalled_openroad_path "/scratch_disks/scratch3/git_runner/deps/OpenROAD/build/bin/openroad"
status=$?
set -e
exit $status

# Examples:
# pytest -q --maxfail=1 --disable-warnings --junitxml=reports/test-results.xml
# npm test -- --ci
# ctest --test-dir build --output-on-failure

# echo "✅ All checks passed."
