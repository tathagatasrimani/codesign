#!/usr/bin/env bash
set -e

# PROJECT ROOT (one level up from setup_scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] PROJECT ROOT = $PROJECT_ROOT"

HIDA_ROOT="$PROJECT_ROOT/ScaleHLS-HIDA"

FORCE_FULL=0
for arg in "$@"; do
    [[ "$arg" == "--full" ]] && FORCE_FULL=1
done

if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] FULL BUILD enabled"

    # Clean previous builds
    rm -rf "$HIDA_ROOT/polygeist/build"
    rm -rf "$HIDA_ROOT/polygeist/llvm-project/build"
    rm -rf "$HIDA_ROOT/build"

    # 1. Build fast LLVM
    bash "$PROJECT_ROOT/setup_scripts/build_polygeist_llvm.sh"

    # 2. Build Polygeist frontend
    bash "$PROJECT_ROOT/setup_scripts/build_polygeist_frontend.sh"

    # 3. Build ScaleHLS standalone
    bash "$PROJECT_ROOT/setup_scripts/build_scalehls.sh"
else
    echo "[setup] Skipping rebuild (no --full)"
fi

###############################################################################
# Set up environment variables
###############################################################################

LLVM_HOME="$HIDA_ROOT/polygeist/llvm-project/build"
POLYGEIST_BIN="$HIDA_ROOT/polygeist/build/bin"
SCALEHLS_BIN="$HIDA_ROOT/build/bin"

export LLVM_HOME
export MLIR_HOME="$LLVM_HOME"
export PATH="$LLVM_HOME/bin:$POLYGEIST_BIN:$SCALEHLS_BIN:$PATH"
export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"

echo "[setup] DONE — fast LLVM + cgeist + scalehls-opt active"
echo "[setup] which clang → $(which clang)"
echo "[setup] which mlir-opt → $(which mlir-opt)"
echo "[setup] which cgeist → $(which cgeist)"
echo "[setup] which scalehls-opt → $(which scalehls-opt)"
