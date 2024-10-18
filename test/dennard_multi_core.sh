#!/bin/sh
cd ..

while getopts c:n:f:a:A: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        c) ARCH_CONFIG=${OPTARG};;
        f) SAVEDIR=${OPTARG};;
        t) TEST_TYPE=${OPTARG};;
        A) ARCH_SEARCH_NUM=${OPTARG};;
        a) AREA=${OPTARG};;
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
    if [ $AREA ]; then
        ARGS+=" --area $AREA"
    fi
    if [ $ARCH_SEARCH_NUM ]; then
        ARGS+=" --num_arch_search_iters $ARCH_SEARCH_NUM"
    fi
    echo $ARGS
    python -m test.experiments.dennard_multi_core $ARGS
fi