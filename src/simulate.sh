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
    python3.8 instrument.py $FILEPATH
    python3.8 instrumented_files/xformed-$name > instrumented_files/output.txt
    if [ $QUIET ]; then
        echo "Quiet mode"
        python3.8 simulate.py $FILEPATH --notrace
    else
        python3.8 simulate.py $FILEPATH
    fi
fi
