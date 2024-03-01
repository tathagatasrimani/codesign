#!/bin/sh
while getopts a:sqn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        a) AREA=${OPTARG};;
    esac
done

# arguments like this: ./codesign.sh -n <name>
if [ $name ]; then
    FILEPATH=benchmarks/models/$name
    ARGS=$FILEPATH
    if [ $QUIET ]; then
        ARGS+=" --notrace"
    fi
    if [ $AREA ]; then
        ARGS+=" --area $AREA"
    fi
    echo $ARGS
    python codesign.py $ARGS
fi