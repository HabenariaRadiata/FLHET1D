#!/bin/bash

# Base configuration file
BASE_CONFIG="my_config.ini"

# Voltages to run
VOLTAGES=(250 350 400 450 500 550 650 750 800 850 900 950)

# Number of parallel jobs
MAX_PROCS=12

# Function to run a single instance
run_instance() {
    local voltage=$1
    local temp_config="temp_${voltage}V.ini"

    # Create a temporary config file by replacing lines
    sed -e "s/^Voltage.*/Voltage             = ${voltage}/" \
        -e "s|^Result dir.*|Result dir        = ./Results/test_${voltage}V|" \
        "$BASE_CONFIG" > "$temp_config"

    # Run the python script with the temp config
    python3 FLHET_compiled.py "$temp_config"
}

export -f run_instance
export BASE_CONFIG

# Run all instances in parallel (up to MAX_PROCS at a time)
printf "%s\n" "${VOLTAGES[@]}" | xargs -n 1 -P $MAX_PROCS -I {} bash -c 'run_instance "$@"' _ {}

