#!/bin/sh
while getopts b:c:f:sqn:p: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        c) ARCH_CONFIG=${OPTARG};;
        b) BW=${OPTARG};;
        p) PARASITIC=${OPTARG};;
    esac
done

# arguments like this: ./simulate.sh -n <name>
if [ $name ]; then
    FILEPATH=src/benchmarks/models/$name.py
    cd ..
    python -m src.instrument $FILEPATH
    python -m src.instrumented_files.xformed-$name > src/instrumented_files/output.txt
    ARGS=$FILEPATH
    if [ $QUIET ]; then
        ARGS+=" --notrace"
    fi
    if [ $ARCH_CONFIG ]; then
        ARGS+=" --architecture_config $ARCH_CONFIG"
    fi
    if [ $BW ]; then
        ARGS+=" --bw $BW"
    fi
    if [ $PARASITIC ]; then
        ARGS+=" --parasitic $PARASITIC"
    fi
    echo $ARGS
    python -m src.simulate $ARGS
fi