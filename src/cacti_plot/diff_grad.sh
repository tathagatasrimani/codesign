#!/bin/sh
cd ..

# Define default values for cfg_file and sympy_file
CFG_FILE="cacti/mem_validate_cache.cfg"
SYMPY_FILE="sympy_mem_validate_access_time.txt"

# Parse command-line arguments to override defaults
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -CFG)
        CFG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        -SYMPY)
        SYMPY_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

# Run the Python script with the defined variables
python cacti_plot/diff_eval.py "$CFG_FILE" "$SYMPY_FILE"
