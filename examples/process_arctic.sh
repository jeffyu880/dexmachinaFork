#!/bin/bash

# Process Arctic dataset demonstrations using retargeting script
# Processes each demonstration file with MANO model retargeting

BASE_PATH="/home/jeffrey/Documents/Manipulation/arctic/outputs/processed_verts/seqs"
MANO_MODEL_DIR=/home/jeffrey/Documents/Manipulation/ManipTrans/maniptrans_envs/assets/mano_urdf
DEMOS_FILE="$(dirname "$0")/demonstrations_by_object.txt"

# Convert to zero-padded format (e.g., 1 -> 01, 2 -> 02)
pad_number() {
    printf "%02d" $1
}

# Process each non-comment, non-empty line from demonstrations_by_object.txt
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip comments and blank lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    # Parse: "object  seq:episodes"
    object=$(echo "$line" | awk '{print $1}')
    rest=$(echo "$line" | awk '{print $2}')
    seq="${rest%%:*}"
    episodes="${rest##*:}"

    # Split episodes by comma and process each one
    IFS=',' read -ra ep_array <<< "$episodes"
    for ep in "${ep_array[@]}"; do
        # Remove leading whitespace
        ep=$(echo "$ep" | xargs)

        # Construct filename with zero-padded episode number
        padded_ep=$(pad_number "$ep")
        FNAME="$BASE_PATH/$seq/${object}_use_$padded_ep.npy"
        
        # Check if file exists before processing
        if [ ! -f "$FNAME" ]; then
            echo "⚠️  File not found: $FNAME"
            continue
        fi
        
        echo "Processing: $FNAME"
        
        # Retry loop until "Saved to" is found
        max_attempts=100
        attempt=1
        while [ $attempt -le $max_attempts ]; do
            output=$(cd dexmachina/retargeting && python process_arctic.py -p "$FNAME" -mmd "$MANO_MODEL_DIR" -sd ../assets/arctic/processed --farthest_sample --save 2>&1)
            if echo "$output" | grep -q "Saved to"; then
                echo "$output"
                echo "✓ Completed: $FNAME"
                sleep 10
                break
            else
                echo "Attempt $attempt failed, retrying in 10 seconds..."
                if [ $attempt -lt $max_attempts ]; then
                    sleep 10
                fi
            fi
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            echo "✗ Failed after $max_attempts attempts: $FNAME"
            echo "Last output:"
            echo "$output"
        fi
    done
done < "$DEMOS_FILE"

echo "All demonstrations processed."
