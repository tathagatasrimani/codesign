#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HIDA_ROOT="$PROJECT_ROOT/ScaleHLS-HIDA"

LLVM_BUILD="$HIDA_ROOT/polygeist/llvm-project/build"
POLY_SRC="$HIDA_ROOT/polygeist"
POLY_BUILD="$POLY_SRC/build"

echo "[polygeist-frontend] Building Polygeist frontend"

mkdir -p "$POLY_BUILD"
cd "$POLY_BUILD"

rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake -G Ninja .. \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DCLANG_DIR="$LLVM_BUILD/lib/cmake/clang" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

ninja

echo "[polygeist-frontend] DONE"
