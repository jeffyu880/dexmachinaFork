#!/bin/bash

MAX_ATTEMPTS=20
RETRY_DELAY=15
# retarget_name is either para or pure_ik_para

CK=/home/jeffrey/Documents/Manipulation/Genesis/dexmachina/izar_logs/rl_games/allegro_hand/allegro-new_obs_residual0424_171847_ketchup40-140-s02-u01_B11000_residual_thres0.5_ho32_imi0.2_con2.0_bc0.2/nn/last_allegro_hand_ep_4500_rew_85.38478.pth
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
        -di 0 \
        --retarget_type pure_ik_para \
        --save_traj
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
