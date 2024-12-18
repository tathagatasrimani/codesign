#!/bin/sh
while getopts a:c:f:qn:o:N:A:p: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        q) QUIET=true;;
        a) AREA=${OPTARG};;
        c) ARCH_CONFIG=${OPTARG};;
        f) SAVEDIR=${OPTARG};;
        o) OPT=${OPTARG};;
        N) NUM=${OPTARG};;
        A) ARCH_SEARCH_NUM=${OPTARG};;
        p) PARASITIC=${OPTARG};;
        s) SCHEDULE=${OPTARG};;
    esac
done

# arguments like this: ./codesign.sh -n <name>
if [ $name ]; then
    FILEPATH=src/benchmarks/models/$name.py
    cd ..
    python -m src.instrument $FILEPATH
    python -m src.instrumented_files.xformed-$name > src/instrumented_files/output.txt
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
    if [ $PARASITIC ]; then
        ARGS+=" --parasitic $PARASITIC"
    fi
    if [ $OPT ]; then  # should be scp, ipopt
        ARGS+=" --opt $OPT"
    fi
    if [ $NUM ]; then
        ARGS+=" --num_iters $NUM"
    fi
    if [ $ARCH_SEARCH_NUM ]; then
        ARGS+=" --num_arch_search_iters $ARCH_SEARCH_NUM"
    fi
    if [ $SCHEDULE ]; then
        ARGS+=" --schedule $SCHEDULE"
    fi
    echo $ARGS
    python -m src.codesign $ARGS
fi