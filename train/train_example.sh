#!/bin/bash

# root directory of the project
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

CONFIG="config/spt_base_cfg.json"
AUDIO_DIR="../../../dataset/wav48_silence_trimmed"


# NPROC_PER_NODE=4
# CUDA_VISIBLE_DEVICES=1,2,6,7 torchrun \
#     --nnode 1 \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master_port 50025  \
# train_example.py \
#     --config ${CONFIG} \

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    train_example.py \
    --config "$PROJECT_ROOT/$CONFIG" \
    --audio_dir "$AUDIO_DIR"
