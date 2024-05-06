#!/bin/sh
while getopts b:c:f:sqn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        c) ARCH_CONFIG=${OPTARG};;
        b) BW=${OPTARG};;
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
    if [ $ARCH_CONFIG ]; then
        ARGS+=" --architecture_config $ARCH_CONFIG"
    fi
    if [ $BW ]; then
        ARGS+=" --bw $BW"
    fi
    echo $ARGS
    python simulate.py $ARGS
fi