# Save parameters to a file
PARAM_FILE="training_params_$(date +%Y%m%d_%H%M%S).txt"

# Build the training command once so we can log and execute the exact same args.
CMD=(
    python dexmachina/rl/train_rl_games.py
    -B 2048 -obf -obt --max_epochs 5000
    --actuate_object --retarget_name para --horizon 16
    -imw 0.5 --learning_rate 0.0003
    --gain_mode all --curr_schedule uniform --wait_epochs 100
    --fixed_mode uniform --uniform_mode slow --group_collisions
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1
    --task_rew_betas 10 1 5 --action_penalty 0.01 --dialback_ep_len 80
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01
    --skip_grad --deque_len 30 --save_freq 500 --use_retarget_contact
    -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5
    --clip ketchup-20-120
    -imi 5 -bc 5 -con 3 -ert 0.8 -exp example
)

PRETTY_CMD=$(cat <<'EOF'
python dexmachina/rl/train_rl_games.py \
    -B 2048 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 \
    -imw 0.5 --learning_rate 0.0003 \
    --gain_mode all --curr_schedule uniform --wait_epochs 100 \
    --fixed_mode uniform --uniform_mode slow --group_collisions \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
    --task_rew_betas 10 1 5 --action_penalty 0.01 --dialback_ep_len 80 \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
    --skip_grad --deque_len 30 --save_freq 500 --use_retarget_contact \
    -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --clip ketchup-20-120 \
    -imi 5 -bc 5 -con 3 -ert 0.6 -exp example
EOF
)

# Extract the command and save it
{
    echo "Training Parameters - $(date)"
    echo "==============================================="
    echo ""
    echo "Command executed:"
    echo ""
    echo "$PRETTY_CMD"
} > $PARAM_FILE

"${CMD[@]}"

echo "Parameters saved to: $PARAM_FILE"