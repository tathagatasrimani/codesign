#!/bin/sh
cd ..

# Define default values for cfg_file, sympy_file, and gen_flag
CFG_FILE="cache"
DAT_FILE=""
SYMPY_FILE=""
GEN_FLAG="false"

# Parse command-line arguments to override defaults
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -CFG)
        CFG_FILE="$2"
        shift 
        shift 
        ;;
        -DAT)
        DAT_FILE="$2"
        shift
        shift
        ;;
        -SYMPY)
        SYMPY_FILE="$2"
        shift 
        shift 
        ;;
        -gen)
        GEN_FLAG="true"
        shift 
        ;;
        *) 
        shift 
        ;;
    esac
done

# Call the Python script with the provided or default arguments
python -m src.cacti_validation.abs_eval -CFG "$CFG_FILE" -DAT "$DAT_FILE" -SYMPY "$SYMPY_FILE" -gen "$GEN_FLAG"
python -m src.cacti_validation.plot_abs
