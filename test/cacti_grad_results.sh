### Results found in src/cacti_validation/grad_plots and src/cacti_validation/results

#!/bin/sh
cd ..

# Define default values for cfg_file, sympy_file, and gen_flag
CFG_FILE="base_cache"
DAT_FILE=""
SYMPY_FILE="Buf"
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

# Run the Python script with the defined variables
python -m src.cacti_validation.grad_eval -CFG "$CFG_FILE" -DAT "$DAT_FILE" -SYMPY "$SYMPY_FILE" -gen "$GEN_FLAG"
python -m src.cacti_validation.plot_grad -CFG "$CFG_FILE"