#!/usr/bin/env bash

set -x

EXP_DIR=./run/self_attn_final_exp1
PY_ARGS=${@:1}

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --data_mode '15frames' \
    --num_global_frames 1 \
    --num_support_frames 1 \
    --num_frames 3 \
    --enc_temporal_window 5 \
    --num_feature_levels 4 \
    --batch_size 2 \
    --lr 5e-5 \
    --dist_url tcp://127.0.0.1:50001 \
    --enc_connect_all_frames \
    --shuffled_aug "centerCrop" \
    --with_box_refine \
    --num_workers 12 \
    ${PY_ARGS}
    
# finetune using SGD

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --data_mode '15frames' \
    --num_global_frames 2 \
    --num_feature_levels 4 \
    --batch_size 1 \
    --lr 5e-5 \
    --self_attn \
    --dist_url tcp://127.0.0.1:50001 \
    --shuffled_aug "centerCrop" \
    --resume ./checkpoint.pth \
    --sgd
    ${PY_ARGS}