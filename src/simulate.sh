#!/bin/sh

FILE_DIR=/Users/PatrickMcEwen/high_level_synthesis/venv/codesign/src

if [ $1 ]; then
    cd $FILE_DIR
    python3 instrument.py cfg/$1
    python3 xformed-$2 > output.txt
    python3 new_simulate.py $1
fi