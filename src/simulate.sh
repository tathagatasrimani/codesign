#!/bin/sh
while getopts a:b:sqn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        s) SEARCH=true;;
        a) AREA=${OPTARG};;
        b) BW=${OPTARG};;
    esac
done

# arguments like this: ./simulate.sh -n <name>
if [ $name ]; then
    FILEPATH=benchmarks/models/$name
    python instrument.py $FILEPATH
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    ARGS=$FILEPATH
    echo $SEARCH
    if [ $SEARCH ]; then
        ARGS+=" --archsearch"
    fi
    if [ $QUIET ]; then
        ARGS+=" --notrace"
    fi
    if [ $AREA ]; then
        ARGS+=" --area $AREA"
    fi
    if [ $BW ]; then
        ARGS+=" --bw $BW"
    fi
    echo $ARGS
    python simulate.py $ARGS
fi