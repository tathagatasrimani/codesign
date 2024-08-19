#!/bin/sh
cd ..

# Define default values for cfg_file, sympy_file, tech_param_key, gen_flag, and metric
CFG_FILE="cacti/mem_validate_cache.cfg"
SYMPY_FILE="sympy_mem_validate"
TECH_PARAM_KEY="Vdd"
GEN_FLAG="false"
METRIC="access_time"

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
        -v)
        TECH_PARAM_KEY="$2"
        shift # past argument
        shift # past value
        ;;
        -gen)
        GEN_FLAG="true"
        shift # past argument
        ;;
        -metric)
        METRIC="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

# Run the Python script with the defined variables
python cacti_plot/diff_eval.py -CFG "$CFG_FILE" -SYMPY "$SYMPY_FILE" -v "$TECH_PARAM_KEY" -gen "$GEN_FLAG" -metric "$METRIC"
