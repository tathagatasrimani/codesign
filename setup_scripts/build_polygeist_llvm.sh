#!/usr/bin/env bash
set -e

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# Ensure compilers are set (RHEL needs this)
export CC=clang
export CXX=clang++

cd polygeist/llvm-project
mkdir -p build
cd build

# Remove stale CMake configuration
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake -G "$CMAKE_GENERATOR" \
  ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
else 
  make -j "$(nproc)"
fi
