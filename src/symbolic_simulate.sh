#!/bin/sh
while getopts c:qn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        c) ARCH_CONFIG=${OPTARG};;
        o) OPT=${OPTARG};;
    esac
done

# arguments like this: ./simulate.sh -n <name>
if [ $name ]; then
    FILEPATH=benchmarks/models/$name
    python instrument.py $FILEPATH
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    ARGS=$FILEPATH
    OPT_ARGS=""
    if [ $QUIET ]; then
        ARGS+=" --notrace"
    fi
    if [ $ARCH_CONFIG ]; then
        ARGS+=" --architecture_config $ARCH_CONFIG"
        OPT_ARGS+=" --architecture_config $ARCH_CONFIG"
    fi
    if [ $OPT ]; then  # options scp, ipopt
        OPT_ARGS+=" --opt $OPT"
    fi

    echo $ARGS
    python symbolic_simulate.py $ARGS
    python optimize.py $OPT_ARGS
fi