#!/bin/sh
while getopts a:qn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        a) AREA=${OPTARG};;
    esac
done

# arguments like this: ./simulate.sh -n <name>
if [ $name ]; then
    FILEPATH=benchmarks/models/$name
    python instrument.py $FILEPATH
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    ARGS=$FILEPATH
    if [ $QUIET ]; then
        ARGS+=" --notrace"
    fi
    # if [ $AREA ]; then
    #     ARGS+=" --area $AREA"
    # fi

    echo $ARGS
    python symbolic_simulate.py $ARGS
    python optimize.py
fi