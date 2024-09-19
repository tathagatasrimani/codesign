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
if [ $CFG_FILE ]; then
    ARGS+=" --config $CFG_FILE"
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


# Run the Python script with the defined variables
python -m src.cacti_validation.grad_eval $ARGS
python -m src.cacti_validation.plot_grad $ARGS
