#!/bin/sh

FILE_DIR=/Users/PatrickMcEwen/forward_pass_present_June/codesign/src # change path name for local computer
# arguments like this: ./simulate.sh benchmarks/nonai_models/<name> <name>
if [ $1 ]; then
    cd $FILE_DIR
    python3 instrument.py cfg/$1
    python3 xformed-$2 > output.txt
    python3 new_simulate.py $1
fi