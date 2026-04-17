# Save parameters to a file
PARAM_FILE="training_params_$(date +%Y%m%d_%H%M%S).txt"
TIMESTAMP=$(date +%m%d_%H%M%S)


# Define parameters once as associative arrays
declare -A PARAMS=(
    [batch_size]="-B 12000"      # should be the num_envs
    [epochs]="-obf -obt --max_epochs 5000"
    [object]="--actuate_object --retarget_name para --horizon 32"
    [learning]="-imw 0.5 --learning_rate 0.0003"            # imw -> imi_wrist_weight. Balannces reward between wrist pose and fingertip pose
    [curriculum]="--gain_mode all --curr_schedule uniform --wait_epochs 200 --num_zero_epoch 500"
    [gains]="--fixed_mode uniform --uniform_mode slow --group_collisions"
    [rewards]="--contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.6 0.6 1"
    [task_rewards]="--task_rew_betas 10 1 5 --action_penalty 0.01 --dialback_ep_len 30"     # obj_pos_beta, obj_rot_beta, obj_arti_beta
    [thresholds]="--aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0 0 0"       # rew_thresholds
    [training]="--skip_grad --deque_len 20 --save_freq 500 --use_retarget_contact"
    [arm_model]="-am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5"
    [demo]="--clip ketchup-30-130"
    [weights]="-imi 0.2 -bc 0.2 -con 2.0 -ert 0.5"      # imi_rew_weight, bc_rew_weight, contact_rew_weight, early_reset_thershold
    [experiment]="-exp mandi3_${TIMESTAMP}"
    [hand]="--hand inspire_hand"
    [seed]="--seed 456"
    # [checkpoint]="--checkpoint /home/jeffrey/Documents/Manipulation/Genesis/dexmachina/logs/rl_games/allegro_hand/allegro-example_ketchup20-140-s01-u01_B2048_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/last_allegro_hand_ep_3500_rew_77.97218.pth"
)

# dialback episode length: minimum episode length that triggers dial back (reducing curriculum difficulty)

# Build the training command once so we can log and execute the exact same args.
CMD=(
    python dexmachina/rl/train_rl_games.py
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
    ${PARAMS[demo]}
    ${PARAMS[weights]} 
    ${PARAMS[experiment]}
    ${PARAMS[hand]}
    ${PARAMS[seed]}
    # ${PARAMS[checkpoint]}
)

# Generate PRETTY_CMD by joining parameters with backslashes and proper formatting
PRETTY_CMD="python dexmachina/rl/train_rl_games.py \\"
for key in batch_size epochs object learning curriculum gains rewards task_rewards thresholds training arm_model demo weights experiment hand seed; do
    PRETTY_CMD+="
    ${PARAMS[$key]} \\"
done
# Remove trailing backslash
PRETTY_CMD=${PRETTY_CMD%' \\'}


Extract the command and save it
{
    echo "Training Parameters - $(date)"
    echo "==============================================="
    echo ""
    echo "Command executed:"
    echo ""
    echo "$PRETTY_CMD"
} > $PARAM_FILE

"${CMD[@]}"

# echo "Parameters saved to: $PARAM_FILE"
