#!/bin/bash

# Base configuration file
BASE_CONFIG="my_config.ini"

# Voltages to sweep
VOLTAGES=(250 350 400 450 500 550 650 750 800 850 900 950)

# Number of parallel jobs allowed
MAX_PROCS=12

# Function that modifies the config and runs Python
run_instance() {
    local voltage=$1
    local temp_config="temp2p5mgs_${voltage}V.ini"

    echo "Starting voltage ${voltage}V"

    # Create modified .ini
    sed -e "s/^Voltage.*/Voltage             = ${voltage}/" \
        -e "s|^Result dir.*|Result dir        = ./Results/voltage_sweep3p5mgs/test_${voltage}V|" \
        "$BASE_CONFIG" > "$temp_config"

    # Run program
    python3 FLHET_compiled.py "$temp_config"

    echo "Finished voltage ${voltage}V"
}

export -f run_instance
export BASE_CONFIG

# Launch all jobs in parallel (max $MAX_PROCS at once)
printf "%s\n" "${VOLTAGES[@]}" | xargs -n 1 -P $MAX_PROCS -I {} bash -c 'run_instance "$@"' _ {}

