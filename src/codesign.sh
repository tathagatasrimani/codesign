#!/bin/sh
while getopts a:c:f:qn:o: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        a) AREA=${OPTARG};;
        c) ARCH_CONFIG=${OPTARG};;
        f) SAVEDIR=${OPTARG};;
        o) OPT=${OPTARG};;
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
    if [ $ARCH_CONFIG ]; then
        ARGS+=" --architecture_config $ARCH_CONFIG"
    fi
    if [ $SAVEDIR ]; then
        ARGS+=" --savedir $SAVEDIR"
    fi
    if [ $OPT ]; then  # should be scp, ipopt
        ARGS+=" --opt $OPT"
    fi
    echo $ARGS
    python codesign.py $ARGS
fi