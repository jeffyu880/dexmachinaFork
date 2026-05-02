#!/usr/bin/env bash
set -o pipefail

# Retry configuration
MAX_RETRIES=10
SUCCESS=false

# Error patterns to detect (case-insensitive)
ERROR_PATTERNS=(
    "TypeError"
    "AttributeError"
    "ValueError"
    "RuntimeError"
    "ImportError"
    "ModuleNotFoundError"
    "KeyError"
    "IndexError"
    "FileNotFoundError"
    "OSError"
    "AssertionError"
)

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[Attempt $attempt/$MAX_RETRIES] Running playback script..."

    python examples/playback_kinematic_retargeted_hand.py \
    -lf dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_02_vector_para.pt \
    --hand allegro_hand \
    --obj_name ketchup \
    --vis 2>&1 | tee /tmp/playback_output.log
    
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Playback completed successfully"
        SUCCESS=true
        break
    fi

    echo "✗ Attempt $attempt failed (exit code $EXIT_CODE), retrying in 10 seconds..."
    sleep 10
done

if [ "$SUCCESS" = true ]; then
    echo "✓ Playback script completed successfully"
    exit 0
else
    echo "✗ Playback script failed after $MAX_RETRIES attempts"
    exit 1
fi