#!/usr/bin/env bash
set -o pipefail

# Retry configuration
MAX_RETRIES=10
SUCCESS=false

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[Attempt $attempt/$MAX_RETRIES] Running playback script..."

    python examples/playback_kinematic_retargeted_hand.py \
            --load_fname dexmachina/assets/retargeter_results/allegro_hand/s01/ketchup_use_01_vector.npy \
            --record_video --vis 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Playback completed successfully"
        SUCCESS=true
        break
    fi

    echo "✗ Script failed with exit code $EXIT_CODE"
    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "Retrying in 15 seconds..."
        sleep 15
    fi
done

if [ "$SUCCESS" = true ]; then
    echo "✓ Playback script completed successfully"
    exit 0
else
    echo "✗ Playback script failed after $MAX_RETRIES attempts"
    exit 1
fi
