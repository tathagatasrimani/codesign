#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HIDA_ROOT="$PROJECT_ROOT/ScaleHLS-HIDA"

LLVM_BUILD="$HIDA_ROOT/polygeist/llvm-project/build"
SCALEHLS_BUILD="$HIDA_ROOT/build"

echo "[scalehls] Building ScaleHLS (standalone)"

mkdir -p "$SCALEHLS_BUILD"
cd "$SCALEHLS_BUILD"

rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake -G Ninja "$HIDA_ROOT" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DCLANG_DIR="$LLVM_BUILD/lib/cmake/clang" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_EXTERNAL_PROJECTS="" \
  -DLLVM_EXTERNAL_SCALEHLS_SOURCE_DIR=""

ninja

echo "[scalehls] DONE"
