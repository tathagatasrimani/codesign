#!/bin/bash

# Default values for variables
name=""
config=""

# Parse -n and -c options
while getopts n:c: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        c) config=${OPTARG};;
    esac
done

# Check if both -n and -c are set
if [ -z "$name" ] || [ -z "$config" ]; then
    echo "Usage: $0 -n <name> -c <config>"
    exit 1
fi

# Run each .sh script in the background with the parsed arguments
./run_inverse_walkthrough.sh -n "$name" -c "$config" &
./run_inverse_validation.sh -n "$name" -c "$config" &
./dennard_multi_core.sh -n "$name" -c "$config" &

# Wait for all background processes to complete
wait
