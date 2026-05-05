#!/usr/bin/env bash
set -o pipefail

# Retry configuration
MAX_RETRIES=20
RETRY_COUNT=0
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

# Save parameters to a file
PARAM_FILE="training_params_multi_demo_$(date +%Y%m%d_%H%M%S).txt"

# Define parameters once as associative arrays
declare -A PARAMS=(
    [batch_size]="-B 10"      #  the # num_envs
    [epochs]="-obf -obt --max_epochs 20"
    [object]="--actuate_object --retarget_name para --horizon 32"
    [learning]="-imw 0.5 --learning_rate 0.0003"
    [curriculum]="--gain_mode all --curr_schedule uniform --wait_epochs 200 --num_zero_epoch 500"
    [gains]="--fixed_mode uniform --uniform_mode slow --group_collisions"
    [rewards]="--contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.6 0.6 1"
    [task_rewards]="--task_rew_betas 10 1 5 --action_penalty 0.01 --dialback_ep_len 30"
    [thresholds]="--aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0 0 0"
    [training]="--skip_grad --deque_len 20 --save_freq 500 --use_retarget_contact"
    [arm_model]="-am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5"
    [weights]="-imi 0.2 -bc 0.2 -con 2.0 -ert 0.3"
    [experiment]="-exp multi-demo-3-lr-0.0001"
    [hand]="--hand allegro_hand"
    [seed]="--seed 24"
    [no_object]="-no_obj"                           # only do hand mimicing 
    [randomization]="--rand_init_ratio 0.5"         # start the training 50% of the time from a random frame
    [retarget]='--use_ik_retarget'          # use just IK kinematic retargeting
    # [sampling]="--demo_sampling deterministic"
    # [randomization]="-rand_obs"         # rand_obs is randomizing observations into the robot and object policy
    # [checkpoint]="--checkpoint /path/to/your/checkpoint.pth"
)

# Multiple demo clips for training.
# Format: object-start-end[-subject][-use_clip]
# DEMOS=(
#     # "ketchup-30-130-s01-u01"
#     "ketchup-40-140-s02-u01"
#     "ketchup-27-127-s02-u03"
#     "ketchup-27-127-s02-u04"
#     # "ketchup-25-125-s05-u01"
#     # "ketchup-36-136-s06-u02"
#     # "ketchup-19-119-s07-u02"
#     # "ketchup-31-131-s09-u01"
#     # "ketchup-31-131-s09-u03"
#     # "ketchup-25-125-s09-u04"
#     # "ketchup-34-134-s10-u01"
#     # "ketchup-35-135-s10-u02"
# )

DEMOS=(
    "ketchup-0-200-s01-u01"
    # "ketchup-0-500-s01-u02"
    # "ketchup-0-500-s04-u02"
    # "ketchup-0-500-s05-u01"
    # "ketchup-0-500-s06-u01"
    # "ketchup-0-500-s06-u02"
    # "ketchup-0-500-s07-u02"
    # "ketchup-0-500-s08-u01"
    # "ketchup-0-500-s08-u02"
    # "ketchup-0-500-s08-u04"
    # "ketchup-0-500-s09-u01"
    # # "ketchup-0-500-s09-u02"
    # "ketchup-0-500-s09-u03"
    # "ketchup-0-500-s09-u04"
    # "ketchup-0-500-s10-u01"
    # "ketchup-0-500-s10-u02"
)

# Build the training command once so we can log and execute the exact same args.
CMD=(
    python dexmachina/rl/train_rl_multi_demo.py
    # --vis
    ${PARAMS[batch_size]} ${PARAMS[epochs]}
    ${PARAMS[object]}
    ${PARAMS[learning]}
    ${PARAMS[curriculum]}
    ${PARAMS[gains]}
    ${PARAMS[rewards]}
    ${PARAMS[task_rewards]}
    ${PARAMS[thresholds]}
    ${PARAMS[training]}
    ${PARAMS[arm_model]}
    ${PARAMS[weights]}
    ${PARAMS[experiment]}
    ${PARAMS[hand]}
    ${PARAMS[seed]}
    ${PARAMS[sampling]}
    ${PARAMS[randomization]}
    ${PARAMS[no_object]}
    ${PARAMS[retarget]}
    --clips "${DEMOS[@]}"
    # ${PARAMS[checkpoint]}
)

# Generate PRETTY_CMD for easy logging/copy-paste.
PRETTY_CMD="$(printf '%q ' "${CMD[@]}")"
PRETTY_CMD="${PRETTY_CMD% }"

# # Save parameters to a timestamped file.
# {
#     echo "Training Parameters - $(date)"
#     echo "==============================================="
#     echo ""
#     echo "Command executed:"
#     echo ""
#     echo "$PRETTY_CMD"
#     echo ""
#     echo "Demonstrations (${#DEMOS[@]} total):"
#     for demo in "${DEMOS[@]}"; do
#         echo "- $demo"
#     done
# } > "$PARAM_FILE"
# echo "Parameters saved to: $PARAM_FILE"

echo "=========================================="
echo "Training with Automatic Retry (up to $MAX_RETRIES attempts)"
echo "=========================================="
echo ""

# Retry loop
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "[ATTEMPT $RETRY_COUNT/$MAX_RETRIES] Starting training..."
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run training with live output and capture log for post-run checks
    ATTEMPT_LOG="$(mktemp)"
    "${CMD[@]}" 2>&1 | tee "$ATTEMPT_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    OUTPUT="$(cat "$ATTEMPT_LOG")"
    rm -f "$ATTEMPT_LOG"
    
    # Check for errors
    ERROR_FOUND=false
    for pattern in "${ERROR_PATTERNS[@]}"; do
        if echo "$OUTPUT" | grep -iq "$pattern"; then
            ERROR_FOUND=true
            echo ""
            echo "⚠️  Error detected: $pattern"
            break
        fi
    done
    
    # Check exit code
    if [ $EXIT_CODE -ne 0 ]; then
        ERROR_FOUND=true
        echo ""
        echo "⚠️  Exit code: $EXIT_CODE"
    fi
    
    echo ""
    
    # Success condition
    if [ "$ERROR_FOUND" = false ]; then
        echo "✅ Training completed successfully!"
        SUCCESS=true
        break
    else
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            WAIT_TIME=10
            echo "❌ Retrying in ${WAIT_TIME}s... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
            echo "=========================================="
            echo ""
            sleep $WAIT_TIME
        fi
    fi
done

echo ""
echo "=========================================="
if [ "$SUCCESS" = true ]; then
    echo "✅ SUCCESS: Training completed after $RETRY_COUNT attempts"
    exit 0
else
    echo "❌ FAILED: Training failed after $MAX_RETRIES attempts"
    exit 1
fi
