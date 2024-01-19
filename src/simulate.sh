#!/bin/sh

while getopts qn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q)
        QUIET=true
        ;;
    esac
done

# arguments like this: ./simulate.sh -n <name>
if [ $name ]; then
    FILEPATH=benchmarks/models/$name
    python instrument.py $FILEPATH
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    if [ $QUIET ]; then
        echo "Quiet mode"
        python symbolic_simulate.py $FILEPATH --notrace
    else
        python symbolic_simulate.py $FILEPATH
    fi
    python optimize.py
fi