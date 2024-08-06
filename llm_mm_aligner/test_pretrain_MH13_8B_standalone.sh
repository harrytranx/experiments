#!/bin/bash
# README: run this script inside a compute node, from llm_mm_aligner folder: ./experiments/aws_migration/test_train_torchrun_standalone.sh

ALIGNER_PARENT_DIR=/fsx_0/user/tranx
ALIGNER_DEP_DIR=/fsx_0/user/tranx/llm_mm_aligner/replicated
JSON_CONFIG=${ALIGNER_PARENT_DIR}/llm_mm_aligner/experiments/aws_tranx/pretrain_MH_8B_resume.json

# run this 
PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR} torchrun --standalone --nproc_per_node=8 \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/main.py $JSON_CONFIG