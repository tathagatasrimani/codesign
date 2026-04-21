#!/bin/bash

# This script runs end-of-build regression tests after a full build completes.
# It inherits BUILD_START_TIME from full_env_start_inside.sh to continue
# tracking total elapsed time (build + regression tests).

if [[ -z "$BUILD_START_TIME" ]]; then
    echo "ERROR: BUILD_START_TIME is not set. This script should be called after full_env_start_inside.sh."
    exit 1
fi

echo ""
echo ">>> Running end-of-build regression tests..."

# Re-source the environment in incremental mode to ensure all paths,
# conda, and environment variables are properly set up.
FORCE_FULL=0
set --
source full_env_start_inside.sh

# Run regression tests
python3 -m test.regression_run -l end_of_build_tests/end_of_build_tests.list.yaml -m 8
test_status=$?

if [[ $test_status -ne 0 ]]; then
    echo "BUILD COMPLETED, but failed the test cases."
else
    echo "BUILD AND REGRESSION TESTS COMPLETED SUCCESSFULLY."
fi

# End timer (continuing from the original build start time)
end_time=$(date +%s)
duration=$((end_time - BUILD_START_TIME))
minutes=$((duration / 60))
seconds=$((duration % 60))

printf "\nTotal elapsed time (build + regression tests): %d minutes and %d seconds\n" $minutes $seconds
