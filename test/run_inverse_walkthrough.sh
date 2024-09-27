#!/bin/sh
cd ..

while getopts c:n:f:N: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        c) ARCH_CONFIG=${OPTARG};;
        f) SAVEDIR=${OPTARG};;
        N) NUM_ITERS=${OPTARG};;
    esac
done

if [ $name ]; then
    FILEPATH=src/benchmarks/models/$name.py
    python -m src.instrument $FILEPATH
    python -m src.instrumented_files.xformed-$name > src/instrumented_files/output.txt
    ARGS=$FILEPATH
    if [ $ARCH_CONFIG ]; then
        ARGS+=" --architecture_config $ARCH_CONFIG"
    fi
    if [ $SAVEDIR ]; then
        ARGS+=" --savedir $SAVEDIR"
    fi
    if [ $NUM_ITERS ]; then
        ARGS+=" --num_iters $NUM_ITERS"
    fi
    echo $ARGS
    python -m src.inverse_validation.inverse_walkthrough_example $ARGS
fi