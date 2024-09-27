#!/bin/sh
cd ..

while getopts c:d:s:g flag
do
    case "${flag}" in
        f) LOG_FILE=${OPTARG};;
    esac
done

ARGS=""
if [ $LOG_FILE ]; then
    ARGS+=" --file $LOG_FILE"
fi

# Call the Python script with the provided or default arguments
python -m test.netlist_plot $ARGS
