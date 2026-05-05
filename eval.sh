#!/bin/bash

MAX_ATTEMPTS=20
RETRY_DELAY=15
# retarget_name is either para or pure_ik_para

CK=/home/jeffrey/Documents/Manipulation/Genesis/dexmachina/logs/rl_games/allegro_hand/ketchup_combined_0504_154644/nn/last_allegro_hand_ep_5000_rew_49.279194.pth
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "[Attempt $attempt/$MAX_ATTEMPTS] Running eval script..."
    python dexmachina/rl/eval_rl_games.py \
        -B 1 \
        --checkpoint $CK \
        -v \
        --eval_episodes 1 \
        --output_render \
        --num_envs 2 \
        --show_reference \
        --camera_angle isometric \
        --record_video \
        -di 3 \
        --retarget_type pure_ik_para
        # --reference_clip ketchup-30-130-s01-u01 \
        # --hand allegro_hand
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Script completed successfully on attempt $attempt"
        exit 0
    else
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            echo "[FAILED] Attempt $attempt failed. Waiting $RETRY_DELAY seconds before retry..."
            sleep $RETRY_DELAY
        else
            echo "[FAILED] All $MAX_ATTEMPTS attempts failed."
            exit 1
        fi
    fi
done
