#!/bin/bash
git submodule update --init scalehls
cd scalehls
sed -i 's|git@github\.com:|https://github.com/|g' .gitmodules
git submodule update --init polygeist
cd polygeist
git submodule update --init llvm-project
cd ../..


