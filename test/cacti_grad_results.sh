#!/bin/sh
cd ..

while getopts c:d:s:g flag
do
    case "${flag}" in
        c) CFG_FILE=${OPTARG};;
        d) DAT_FILE=${OPTARG};;
        s) SYMPY_FILE=${OPTARG};;
        g) GEN=true;;
    esac
done

ARGS=""
PLOT_ARGS=""
if [ $CFG_FILE ]; then
    ARGS+=" --config $CFG_FILE"
    PLOT_ARGS+=" --config $CFG_FILE"
fi
if [ $DAT_FILE ]; then
    ARGS+=" --dat $DAT_FILE"
fi
if [ $SYMPY_FILE ]; then
    ARGS+=" --sympy $SYMPY_FILE"
fi
if [ $GEN ]; then
    ARGS+=" --gen"
fi


# Run the Python script with the provided arguments
python -m test.grad_eval $ARGS
python -m test.plot_grad $PLOT_ARGS
