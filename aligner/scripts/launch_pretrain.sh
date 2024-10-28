#!/bin/bash
# Usage: ./launch_pretrain.sh
# ./launch_pretrain.sh && sleep 3 && slog
# This is an example launch script for pretrain job. 
# However, users can customize the parameters for different enviroments and workflows (e.g., pretrain, eval, sft)

JOB_NAME=s2_504
NUM_NODES=64
# REPO_LOC=/fsx_0/user/tranx/rsync
REPO_LOC=/fsx_0/user/tranx
JSON_CONFIG=/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/mm9_stage2/stage2_MM9_70B_MH19_504px_64nodes_exp28.json

# JOB_NAME=s2_i18n
# NUM_NODES=64
# REPO_LOC=/fsx_0/user/tranx/rsync
# # JSON_CONFIG=/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/mm9_stage2/stage2_MM9_70B_MH19_336px_2nodes.json
# JSON_CONFIG=/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/mm9_stage2/stage2_MM9_70B_MH19_336px_64nodes_i18n.json

# ======================================
# JOB_NAME=sample
# NUM_NODES=1
# REPO_LOC=/fsx_0/user/tranx/rsync 
# JSON_DIR=$REPO_LOC/llm_mm_aligner/experiments/aws/mm9_stage1
# JSON_CONFIG=$JSON_DIR/pretrain_MM9_70B_MH19_336px_128nodes_bz32_scratch.json

# ======================================
# llm_mm_aligner github
# JOB_NAME=sample
# NUM_NODES=2
# REPO_LOC=/fsx_0/user/tranx # parent dir of llm_mm_aligner
# JSON_CONFIG=/fsx_0/user/tranx/experiments/llm_mm_aligner/stage2_mm9/stage2_MM9_70B_MH19_336px_128nodes.json


# ======================================
RUN_MODULE=$REPO_LOC/llm_mm_aligner/main.py
CONDA_ENV_PATH=/opt/hpcaas/.mounts/fs-036153e63d56f4dc2/home/tranx/conda/envs/aligner_v7 # include full path

JOB_ID=$( \
    sbatch --parsable --job-name=$JOB_NAME --nodes=$NUM_NODES --ntasks=$NUM_NODES sbatch_h100s.sh \
    $RUN_MODULE \
    $JSON_CONFIG \
    $CONDA_ENV_PATH \
    $REPO_LOC \
)

echo $JOB_ID