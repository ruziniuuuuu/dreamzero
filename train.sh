#!/bin/bash
# DreamZero DROID Training Script
#
# Usage:
#   # Set your dataset path and output directory, then run:
#   bash scripts/train/droid_training.sh
#
# Prerequisites:
#   - DROID dataset in LeRobot format at DROID_DATA_ROOT
#   - Model weights auto-download from HuggingFace (Wan-AI/Wan2.1-I2V-14B-480P)
#   - Tokenizer auto-downloads from HuggingFace (google/umt5-xxl)

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Set these to your paths:
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"/mnt/aws-lfs-02/shared/datasets/droid_101_success_idlefiltered6"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid"}
NUM_GPUS=${NUM_GPUS:-4}
# =============================================

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_horizon_relative \
    wandb_project=dreamzero \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_efficient_weighted \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=10 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=440 \
    save_strategy=no \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=/mnt/aws-lfs-02/shared/ckpts/Wan2.1-I2V-14B-480P \
    text_encoder_pretrained_path=/mnt/aws-lfs-02/shared/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=/mnt/aws-lfs-02/shared/ckpts/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=/mnt/aws-lfs-02/shared/ckpts/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    tokenizer_path=/mnt/aws-lfs-02/shared/ckpts/umt5-xxl
