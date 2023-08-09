#!/bin/sh

FILE_DIR=/Users/PatrickMcEwen/git_container/codesign/src # change path name for local computer
# arguments like this: ./simulate.sh benchmarks/models/<name> <name>
if [ $1 ]; then
    cd $FILE_DIR
    python3 instrument.py $1
    python3 instrumented_files/xformed-$2 > instrumented_files/output.txt
    python3 new_simulate.py $1 $3
    #cd destiny/config
    #./destiny sample.cfg
fi