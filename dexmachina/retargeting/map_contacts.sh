#!/bin/bash

# Map contacts script for Arctic dataset
# Runs contact mapping for each processed demonstration

# Configuration
HAND="${1:-allegro_hand}"  # Options: allegro_hand, xhand, inspire_hand, schunk_hand
OBJ="ketchup"

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
    "s01:2"
    # "s02:1,2,3,4"
    # "s04:2"
    # "s05:1"
    # "s06:1,2"
    # "s07:2"
    # "s08:1,2,3,4"
    # "s09:1,2,3,4"
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
        
        # Construct FNAME: assets/arctic/processed/{subject}/{object}_use_{episode}.npy
        FNAME="assets/arctic/processed/${seq}/${OBJ}_use_${padded_user}.npy"
        
        echo "=========================================="
        echo "Processing: $FNAME with hand: $HAND"
        echo "=========================================="
        
        # Check if file exists
        if [ ! -f "$FNAME" ]; then
            echo "⚠️  File not found: $FNAME"
            echo "Skipping..."
            echo ""
            continue
        fi
        
        # Run map contacts
        python retargeting/map_contacts.py \
            --hand "$HAND" \
            --load_fname "$FNAME"
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $FNAME"
        else
            echo "✗ Failed: $FNAME"
        fi
        
        echo "Waiting 10 seconds before next run..."
        sleep 10
        echo ""
    done
done

echo "All demonstrations processed."
