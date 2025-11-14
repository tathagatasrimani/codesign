#!/bin/bash

# Script to compare C++ and Python DESTINY across multiple configurations

CPP_DIR="/Users/aaravwattal/Documents/Stanford/Junior Fall/Memory Modeling RSG/cacti_destiny_old/destiny_3d_cache-master"
PYTHON_DIR="/Users/aaravwattal/Documents/Stanford/Junior Fall/Memory Modeling RSG/cacti_destiny_old/destiny_3d_cache_python"

# Test configurations
configs=(
    "sample_SRAM_2layer"
    "sample_SRAM_4layer"
    "sample_STTRAM"
    "sample_PCRAM"
    "sample_2D_eDRAM"
)

echo "================================================================"
echo "DESTINY C++ vs Python Comparison Test"
echo "================================================================"
echo ""

for config in "${configs[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Testing: $config"
    echo "----------------------------------------------------------------"

    # Run C++ version
    echo "Running C++ DESTINY..."
    cd "$CPP_DIR"
    ./destiny "config/${config}.cfg" -o "cpp_${config}_output.txt" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ C++ completed"
    else
        echo "✗ C++ failed"
    fi

    # Run Python version
    echo "Running Python DESTINY..."
    cd "$PYTHON_DIR"
    python main.py -config "config/${config}.cfg" -output "python_${config}_output.txt" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Python completed"
    else
        echo "✗ Python failed"
    fi

    echo ""
done

echo "================================================================"
echo "All tests complete!"
echo "================================================================"
