#!/bin/bash

# Specify the filename
filename="config_SPT.ini"

generate_config_and_run() {
    local indiceX=$1
    echo $indiceX
    
    debit=$(awk "BEGIN { printf \"%.7f\", $indiceX / 3. }")
    xx=$(awk "BEGIN { printf \"%.7f\", $debit * 1.e-6 }")
    # coeff = 0.0095047 * xx**3 -0.18111132 * xx**2 + 1.16961328 *xx -1.65495335
    # coeff=$(awk "BEGIN { printf \"%.7f\", 0.0082243 * $debit^3 - 0.15413333 * $debit^2 + 0.97893835 * $debit - 1.1749532 }")
    # echo "coeff = $coeff"
    # yy=$(awk "BEGIN { printf \"%.7f\", 250 + $coeff * $debit * 10. }")

    # Define the output filename for each iteration
    out_filename="Results_SPT_h03_1/config_modif_cp_${indiceX}.ini"
    
    # Use sed to replace the lines in the original file and save to the output file
    sed -e "s/Result dir        = \.\/Results_SPT_h03_1\/zzz/Result dir        = .\/Results_SPT_h03_1\/test_${indiceX}/g" "$filename" > "$out_filename"
    sed -i "s/xxx/$xx/g" "$out_filename"
    # sed -i "s/yyy/$yy/g" "$out_filename"
    
    echo "Replacement complete for indices X: $indiceX. Modified content saved to $out_filename."
    python FLHET_compiled.py "$out_filename"
}

# Iterate over indices and launch tasks in the background
for indiceX in $(seq 6 24); do
    generate_config_and_run $indiceX &
    
    # Limit the number of background jobs to avoid overloading the system
   while (( $(jobs | wc -l) >= 4 )); do
       sleep 1
   done
done

# Wait for all background jobs to finish
wait

