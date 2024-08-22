#!/bin/bash

# JSON_CONFIG=/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/pretrain/pretrain_8B_Llama3_336px.json
# ALIGNER_PARENT_DIR=/fsx_0/user/tranx

# JSON_CONFIG=/fsx_0/user/tranx/llm_mm_aligner/experiments/aws_tranx/mm9_stage1/pretrain_MH_8B.json
    # "output_dir": "/fsx_0/checkpoints/tranx/Aligner-Pretrain-8B/output_n2_retrain",
        # "output_dir": "/fsx_0/checkpoints/tmp/pretrain_8B_Llama3_336px",
JSON_CONFIG=pretrain_MH_8B.json
ALIGNER_PARENT_DIR=/fsx_0/user/tranx/rsync
ALIGNER_DEP_DIR=$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated
RUN_MODULE=main.py

echo "Run module: $ALIGNER_PARENT_DIR/llm_mm_aligner/$RUN_MODULE"
echo "Config: $JSON_CONFIG"

env_name=aligner_v7
if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
    eval "$(conda shell.bash hook)"
    conda activate $env_name 
fi

echo Using conda environment: $CONDA_DEFAULT_ENV
if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
    echo "Error: CONDA_DEFAULT_ENV is not set to $env_name"
    exit 1
fi



PYTHON_PATH_EXTRAS=$PYTHONPATH:\
$ALIGNER_PARENT_DIR:\
$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated:\
$CONDA_PREFIX/python-packages

PYTHONPATH=${PYTHONPATH}:${PYTHON_PATH_EXTRAS} torchrun --standalone \
    --nproc_per_node=8 ${ALIGNER_PARENT_DIR}/llm_mm_aligner/$RUN_MODULE $JSON_CONFIG

echo "DONE"
