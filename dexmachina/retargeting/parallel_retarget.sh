#!/bin/bash

# Parallel retargeting script for Arctic dataset
# Runs retargeting for each demonstration clip

# Configuration
HAND="${1:-allegro_hand}"  # Options: allegro_hand, xhand, inspire_hand, schunk_hand
OBJ="ketchup"
CONTROL_STEPS=2000
SAVE_NAME="para"

# Validate hand option
valid_hands=("allegro_hand" "xhand" "inspire_hand" "schunk_hand")
if [[ ! " ${valid_hands[@]} " =~ " ${HAND} " ]]; then
    echo "Invalid HAND option: $HAND"
    echo "Valid options: ${valid_hands[*]}"
    exit 1
fi

echo "Using hand: $HAND"

# Array of demonstrations with their sequences and user numbers
demonstrations=(
    # "s01:2"
    "s02:1"
    # "s04:2"
    # "s05:1"
    # "s06:1,2"
    # "s07:2"
    # "s08:3"
    # "s09:2"
    # "s10:1,2"
)

# Convert to zero-padded format (e.g., 1 -> 01, 2 -> 02)
pad_number() {
    printf "%02d" $1
}

# Process each demonstration
for demo in "${demonstrations[@]}"; do
    # Split sequence and users (e.g., "s02:1,2,3,4" -> seq="s02", users="1,2,3,4")
    seq="${demo%%:*}"
    users="${demo##*:}"
    
    # Split users by comma and process each one
    IFS=',' read -ra user_array <<< "$users"
    for user in "${user_array[@]}"; do
        # Remove leading whitespace
        user=$(echo "$user" | xargs)
        
        # Construct zero-padded user number
        padded_user=$(pad_number "$user")
        
        # Construct CLIP in format: object-0-600-subject-user
        CLIP="${OBJ}-0-600-${seq}-u${padded_user}"
        
        echo "=========================================="
        echo "Processing: $CLIP with hand: $HAND"
        echo "=========================================="
        
        # Retry loop up to 20 times until "Saved data to" is found
        max_attempts=50
        attempt=1
        success=false
        
        while [ $attempt -le $max_attempts ]; do
            echo "Attempt $attempt/$max_attempts"
            
            # Stream output live to console and also capture it for success detection.
            tmp_output=$(mktemp)
            python retargeting/parallel_retarget.py \
                --clip "$CLIP" \
                --hand "$HAND" \
                --control_steps "$CONTROL_STEPS" \
                --save_name "$SAVE_NAME" \
                --save \
                -ow 2>&1 | tee "$tmp_output"
            cmd_status=${PIPESTATUS[0]}
            output=$(cat "$tmp_output")
            rm -f "$tmp_output"

            if [ $cmd_status -ne 0 ]; then
                echo "Python command exited with status: $cmd_status"
            fi
            
            if echo "$output" | grep -q "Saved data to"; then
                echo "✓ Completed: $CLIP"
                success=true
                break
            else
                echo "✗ Attempt $attempt failed: \"Saved data to\" not found"
                if [ $attempt -lt $max_attempts ]; then
                    echo "Waiting 10 seconds before retry..."
                    sleep 10
                fi
            fi
            
            attempt=$((attempt + 1))
        done
        
        if [ "$success" = false ]; then
            echo "✗ Failed after $max_attempts attempts: $CLIP"
        fi
        
        echo "Waiting 10 seconds before next demonstration..."
        sleep 10
        echo ""
    done
done

echo "All demonstrations processed."
