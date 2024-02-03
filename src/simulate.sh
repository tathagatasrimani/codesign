#!/bin/sh
while getopts sqn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        s) SEARCH=true;;
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
    echo $ARGS
    python simulate.py $ARGS
fi