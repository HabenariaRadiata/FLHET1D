#!/bin/bash

# Specify the filename
filename="config_vary_Tecath.ini"

generate_config_and_run() {
    local indiceX=$1
    echo $indiceX
    
    # Define the output filename for each iteration
    out_filename="Results_SPT_2/config_modif_cp_${indiceX}.ini"
    
    # Use sed to replace the lines in the original file and save to the output file
    sed -e "s/Result dir        = \.\/Results_SPT_2\/zzz/Result dir        = .\/Results_SPT_2\/test_${indiceX}/g" "$filename" > "$out_filename"
    sed -i "s/xxx/$indiceX/g" "$out_filename"
    
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

