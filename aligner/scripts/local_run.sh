#!/bin/bash

JSON_CONFIG=$1
ALIGNER_PARENT_DIR=/fsx_0/user/tranx
ALIGNER_DEP_DIR=$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated
RUN_MODULE=main.py

env_name=aligner_v7
if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
    eval "$(conda shell.bash hook)"
    conda activate $env_name 
fi

echo Using conda environment: $CONDA_DEFAULT_ENV
if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
    echo "Error: CONDA_DEFAULT_ENV is not set to $env_name"
    exit 1

# run this 
PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR} torchrun --standalone --nproc_per_node=8 \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/$RUN_MODULE $JSON_CONFIG