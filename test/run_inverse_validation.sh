#!/bin/sh
cd ..

while getopts c:n:f:t: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        c) ARCH_CONFIG=${OPTARG};;
        f) SAVEDIR=${OPTARG};;
        t) TEST_TYPE=${OPTARG};;
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
    if [ $TEST_TYPE ]; then
        ARGS+=" --test_type $TEST_TYPE"
    fi
    echo $ARGS
    python -m test.inverse_validation.inverse_validate $ARGS
fi