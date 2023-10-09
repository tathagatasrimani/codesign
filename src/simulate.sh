#!/bin/sh

while getopts d:n:f: flag
do
    case "${flag}" in
        f) filepath=${OPTARG};; # $1
        n) name=${OPTARG};;
        d) debug=${OPTARG};;
    esac
done

# arguments like this: ./simulate.sh -f benchmarks/models/<name> -n <name>
if [ $filepath ]; then

    python instrument.py $filepath
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    if [ $debug ]; then
        python simulate.py $filepath --notrace
    else
        python simulate.py $filepath
    fi
    # python simulate.py $filepath --debug $debug
    #cd destiny/config
    #./destiny sample.cfg
fi