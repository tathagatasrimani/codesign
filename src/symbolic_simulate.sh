#!/bin/sh
while getopts c:qn: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        c) ARCH_CONFIG=${OPTARG};;
        o) OPT=${OPTARG};;
        s) SCHEDULE=${OPTARG};;
    esac
done

# arguments like this: ./simulate.sh -n <name>
if [ $name ]; then
    FILEPATH=src/benchmarks/models/$name.py
    cd ..
    python -m src.instrument $FILEPATH
    python -m src.instrumented_files.xformed-$name > src/instrumented_files/output.txt
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
    if [ $SCHEDULE ]; then
        ARGS+=" --schedule $SCHEDULE"
    fi
    echo $ARGS
    python -m src.symbolic_simulate $ARGS
    python -m src.optimize $OPT_ARGS
fi