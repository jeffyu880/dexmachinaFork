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

    python examples/playback_kinematic_retargeted_hand.py --obj_name ketchup --record_video --vis --frames 30-130 2>&1
    EXIT_CODE=$?

    # Check if command succeeded
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Playback completed successfully"
        SUCCESS=true
        break
    fi

    # Check for error patterns in output
    ERROR_FOUND=false
    for pattern in "${ERROR_PATTERNS[@]}"; do
        if grep -qi "$pattern" /tmp/playback_output.log; then
            echo "✗ Error detected: $pattern"
            ERROR_FOUND=true
            break
        fi
    done

    if [ "$ERROR_FOUND" = true ] && [ $attempt -lt $MAX_RETRIES ]; then
        echo "Retrying in 15 seconds..."
        sleep 15
        continue
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "✗ Script failed with exit code $EXIT_CODE"
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "Retrying..."
            sleep 15
        fi
    else
        SUCCESS=true
        break
    fi
done

if [ "$SUCCESS" = true ]; then
    echo "✓ Playback script completed successfully"
    exit 0
else
    echo "✗ Playback script failed after $MAX_RETRIES attempts"
    exit 1
fi