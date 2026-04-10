#!/bin/bash

# Process Arctic dataset demonstrations using retargeting script
# Processes each demonstration file with MANO model retargeting

BASE_PATH="/home/jeffrey/Documents/Manipulation/arctic/outputs/processed_verts/seqs"
MANO_MODEL_DIR=/home/jeffrey/Documents/Manipulation/ManipTrans/maniptrans_envs/assets/mano_urdf

# Array of demonstrations with their sequences and episode numbers
demonstrations=(
    # "s01:2"
    "s02:3,4"
    "s04:2"
    "s05:1"
    "s06:1,2"
    "s07:2"
    "s08:1,2,3,4"
    "s09:1,2,3,4"
    "s10:1,2"
)

# Convert to zero-padded format (e.g., 1 -> 01, 2 -> 02)
pad_number() {
    printf "%02d" $1
}

# Process each demonstration
for demo in "${demonstrations[@]}"; do
    # Split sequence and episodes (e.g., "s01:1,2" -> seq="s01", episodes="1,2")
    seq="${demo%%:*}"
    episodes="${demo##*:}"
    
    # Split episodes by comma and process each one
    IFS=',' read -ra ep_array <<< "$episodes"
    for ep in "${ep_array[@]}"; do
        # Remove leading whitespace
        ep=$(echo "$ep" | xargs)
        
        # Construct filename with zero-padded episode number
        padded_ep=$(pad_number "$ep")
        FNAME="$BASE_PATH/$seq/ketchup_use_$padded_ep.npy"
        
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
done

echo "All demonstrations processed."
