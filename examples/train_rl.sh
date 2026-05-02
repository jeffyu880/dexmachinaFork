# Save parameters to a file
PARAM_FILE="training_params_$(date +%Y%m%d_%H%M%S).txt"

MAX_ATTEMPTS=20
RETRY_DELAY=10


for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "[Attempt $attempt/$MAX_ATTEMPTS] Running eval script..."

# Define parameters once as associative arrays
declare -A PARAMS=(
    [batch_size]="-B 2"      # should be the num_envs
    [epochs]="-obf -obt --max_epochs 2"
    [object]="--actuate_object --retarget_name para --horizon 32"       # horizon is the number of env steps collected per actor before each PPO update cycle
    [learning]="-imw 0.5 --learning_rate 0.0003"
    [curriculum]="--gain_mode all --curr_schedule uniform --wait_epochs 200 --num_zero_epoch 500"
    [gains]="--fixed_mode uniform --uniform_mode slow --group_collisions"
    [rewards]="--contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.6 0.6 1"
    [task_rewards]="--task_rew_betas 10 1 5 --action_penalty 0.01 --dialback_ep_len 30"
    [thresholds]="--aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0 0 0"
    [training]="--skip_grad --deque_len 20 --save_freq 500 --use_retarget_contact"
    [arm_model]="-am hybrid --hybrid_scales 0.1 1.0 --kp_init 100 --kv_init 5"
    [demo]="--clip ketchup-30-130-s01-u01"
    [weights]="-imi 0.2 -bc 0.2 -con 2.0 -ert 0.5"
    [experiment]="-exp final_obs"
    [hand]="--hand allegro_hand"
    [seed]="--seed 24"
    # [checkpoint]="--checkpoint /home/jeffrey/Documents/Manipulation/Genesis/dexmachina/izar_logs/rl_games/allegro_hand/allegro-multi_GPU_0412_225545_ketchup40-140-s02-u01_B10000_hybrid_thres0.5_ho32_imi0.2_con2.0_bc0.2/nn/last_allegro_hand_ep_4510_rew__92.501335_.pth"
)

# Build the training command once so we can log and execute the exact same args.
CMD=(
    python dexmachina/rl/train_rl_games.py
    --vis
    ${PARAMS[batch_size]} 
    ${PARAMS[epochs]}
    ${PARAMS[object]}
    ${PARAMS[learning]}
    ${PARAMS[curriculum]}
    ${PARAMS[gains]}
    ${PARAMS[rewards]}
    ${PARAMS[task_rewards]}
    ${PARAMS[thresholds]}
    ${PARAMS[training]}
    ${PARAMS[arm_model]}
    ${PARAMS[demo]}
    ${PARAMS[weights]} 
    ${PARAMS[experiment]}
    ${PARAMS[hand]}
    ${PARAMS[seed]}
    ${PARAMS[checkpoint]}
)

# Generate PRETTY_CMD by joining parameters with backslashes and proper formatting
PRETTY_CMD="python dexmachina/rl/train_rl_games.py \\"
for key in batch_size epochs object learning curriculum gains rewards task_rewards thresholds training arm_model demo weights experiment hand seed; do
    PRETTY_CMD+="
    ${PARAMS[$key]} \\"
done
# Remove trailing backslash
PRETTY_CMD=${PRETTY_CMD%' \\'}


# Extract the command and save it
# {
#     echo "Training Parameters - $(date)"
#     echo "==============================================="
#     echo ""
#     echo "Command executed:"
#     echo ""
#     echo "$PRETTY_CMD"
# } > $PARAM_FILE

    echo "Executing command..."
    "${CMD[@]}"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] Training completed successfully on attempt $attempt"
        exit 0
    else
        echo "[FAILED] Attempt $attempt failed with exit code $EXIT_CODE"
        
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            echo "Retrying in $RETRY_DELAY seconds... (Attempt $((attempt + 1))/$MAX_ATTEMPTS)"
            sleep $RETRY_DELAY
        else
            echo "[ERROR] All $MAX_ATTEMPTS attempts failed. Exiting."
            exit 1
        fi
    fi
done

# echo "Parameters saved to: $PARAM_FILE"