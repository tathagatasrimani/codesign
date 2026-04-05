#!/bin/bash

# Wrapper for export-IP-only setup.
# Mirrors the behavior of full_env_start.sh:
# - If sourced without --full, source the inside script directly.
# - Otherwise, apply taskset core restriction and execute the inside script.

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
    source full_env_start_FPGA_inside.sh "$@"
    return 0
fi

USE_MAX_PARALLEL=0
MAX_PARALLEL_CORES=24

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

LAST_CORE=$((TARGET_CORES - 1))
CPU_LIST="0-${LAST_CORE}"

echo "Restricting export-IP setup to $TARGET_CORES core(s) (taskset -c $CPU_LIST)"
exec taskset -c "$CPU_LIST" bash full_env_start_FPGA_inside.sh "$@"