#!/bin/bash

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

CONFIG="config/spt_base_cfg.json"
AUDIO_DIR="D:/dataset/VCTK-Corpus-0.92/wav48_silence_trimmed"


# NPROC_PER_NODE=4
# CUDA_VISIBLE_DEVICES=1,2,6,7 torchrun \
#     --nnode 1 \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master_port 50025  \
# train_example.py \
#     --config ${CONFIG} \

accelerate launch train_example.py\
    --config "$PROJECT_ROOT/${CONFIG}"\
    --audio_dir ${AUDIO_DIR}\
