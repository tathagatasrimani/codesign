#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HIDA_ROOT="$PROJECT_ROOT/ScaleHLS-HIDA"

LLVM_SRC="$HIDA_ROOT/polygeist/llvm-project"
LLVM_BUILD="$LLVM_SRC/build"

echo "[polygeist-llvm] Building FAST LLVM in: $LLVM_BUILD"

mkdir -p "$LLVM_BUILD"
cd "$LLVM_BUILD"

rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

ninja

echo "[polygeist-llvm] DONE"
