#!/bin/sh
cd ..

while getopts c:n:f: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        c) ARCH_CONFIG=${OPTARG};;
        f) SAVEDIR=${OPTARG};;
    esac
done

if [ $name ]; then
    FILEPATH=benchmarks/models/$name
    python instrument.py $FILEPATH
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    ARGS=$FILEPATH
    if [ $ARCH_CONFIG ]; then
        ARGS+=" --architecture_config $ARCH_CONFIG"
    fi
    if [ $SAVEDIR ]; then
        ARGS+=" --savedir $SAVEDIR"
    fi
    echo $ARGS
    python inverse_validation/inverse_validate.py $ARGS
fi