# Save parameters to a file
PARAM_FILE="training_params_$(date +%Y%m%d_%H%M%S).txt"
TIMESTAMP=$(date +%m%d_%H%M%S)


# Define parameters once as associative arrays
declare -A PARAMS=(
    [batch_size]="-B 15000"      # should be the num_envs
    [epochs]="-obf -obt --max_epochs 5000"
    [object]="--actuate_object --retarget_name para --horizon 32"
    [learning]="-imw 0.5 --learning_rate 0.0003"
    [curriculum]="--gain_mode all --curr_schedule uniform --wait_epochs 200 --num_zero_epoch 500"
    [gains]="--fixed_mode uniform --uniform_mode slow --group_collisions"
    [rewards]="--contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.6 0.6 1"
    [task_rewards]="--task_rew_betas 10 1 5 --action_penalty 0.01 --dialback_ep_len 30"
    [thresholds]="--aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0 0 0"
    [training]="--skip_grad --deque_len 20 --save_freq 500 --use_retarget_contact"
    [arm_model]="-am residual --hybrid_scales 0.1 1.0 --kp_init 100 --kv_init 5"
    [weights]="-imi 0.2 -bc 0.2 -con 2.0 -ert 0.3"
    [experiment]="-exp residual"
    [hand]="--hand allegro_hand"
    [seed]="--seed 24"
    # [no_object]="-no_obj"                           # only do hand mimicing 
    [randomization]="--rand_init_ratio 0.5"         # start the training 50% of the time from a random frame
    [retarget]='--use_ik_retarget'          # use just IK kinematic retargeting
    # [sampling]="--demo_sampling deterministic"
    # [randomization]="--use_rand --rand_friction --rand_com --rand_mass"
    # [checkpoint]="--checkpoint /path/to/your/checkpoint.pth"
    [residual_cap]="--res_cap"      # cap the max displacement and rotation away from inital pose for wrist
    # [action_smoothing]="--action_moving_avg 0.8"    # the most recent action is a combination of the current and the previous 
)

DEMOS=(
    "ketchup-30-130-s01-u01"            # DONT START FROM STEP 0
    # "ketchup-20-120-s01-u02"
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
    ${PARAMS[randomization]}
    ${PARAMS[no_object]}
    ${PARAMS[retarget]}
    ${PARAMS[residual_cap]}
    ${PARAMS[action_smoothing]}
    # ${PARAMS[checkpoint]}
    --clips "${DEMOS[@]}"
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

"${CMD[@]}"

# echo "Parameters saved to: $PARAM_FILE"
