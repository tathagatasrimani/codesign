#!/bin/bash

# This wrapper determines the allowed CPU cores and uses taskset to
# restrict full_env_start.sh to that many cores.

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    IS_SOURCED=1
fi

HAS_FULL_FLAG=0
for arg in "$@"; do
    if [[ "$arg" == "--full" ]]; then
        HAS_FULL_FLAG=1
        break
    fi
done

if [[ $IS_SOURCED -eq 1 && $HAS_FULL_FLAG -eq 0 ]]; then
    source full_env_start_inside.sh "$@"
    return 0
fi

USE_MAX_PARALLEL=0
MAX_PARALLEL_CORES=24

# Parse command line options (mirror the same flags as full_env_start.sh)
for arg in "$@"; do
    if [[ "$arg" == "--max_parallel_install" ]]; then
        USE_MAX_PARALLEL=1
    fi
done

TOTAL_CORES=$(nproc 2>/dev/null || echo 0)
if [[ $TOTAL_CORES -le 0 ]]; then
    TOTAL_CORES=1
fi

if [[ $USE_MAX_PARALLEL -eq 1 ]]; then
    TARGET_CORES=$TOTAL_CORES
else
    TARGET_CORES=$((TOTAL_CORES / 2))
    if [[ $TARGET_CORES -lt 1 ]]; then
        TARGET_CORES=1
    fi
fi

if [[ $TARGET_CORES -gt $MAX_PARALLEL_CORES ]]; then
    TARGET_CORES=$MAX_PARALLEL_CORES
fi

# Build a CPU list 0-(TARGET_CORES-1) for taskset
LAST_CORE=$((TARGET_CORES - 1))
CPU_LIST="0-${LAST_CORE}"

echo "Restricting build to $TARGET_CORES core(s) (taskset -c $CPU_LIST)"
exec taskset -c "$CPU_LIST" bash full_env_start_inside.sh "$@"
