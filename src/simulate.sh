#!/bin/sh
SEARCH=false
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

    if [ $QUIET ]; then
        echo "Quiet mode"
        python simulate.py $FILEPATH --notrace --archsearch $SEARCH
    else
        python simulate.py $FILEPATH --notrace --archsearch $SEARCH
    fi
fi