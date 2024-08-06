#!/bin/bash
# USAGE:
# Accquire a compute node with 8 GPUs
# conda activate aligner_v7
# ibatch llm_mm_aligner/experiments/aws_tranx/test_pretrain_MH13_70B_standalone.sh

# export NCCL_IB_TIMEOUT=30
export LOGLEVEL=INFO
export CUDA_LAUNCH_BLOCKING=0
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_BUFFSIZE=8388608 
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_DEBUG=DEBUG
export NCCL_SOCKET_IFNAME="enp"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker0,nerdctl0,veth*"
export NCCL_IB_DISABLE=1
export CUDA_CACHE_PATH="/fsx_0/user/${USER}/.nv/ComputeCache"
export LD_PRELOAD=/usr/local/cuda-12.3/lib/libnccl.so
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH

CONDA_ENV=aligner_v7
ALIGNER_PARENT_DIR=/fsx_0/user/tranx
ALIGNER_DEP_DIR=/fsx_0/user/tranx/llm_mm_aligner/replicated
CONDA_PYTHON_PKGS=/data/home/tranx/conda/envs/$CONDA_ENV/python-packages

# JSON_CONFIG=${ALIGNER_PARENT_DIR}/llm_mm_aligner/experiments/aws_tranx/pretrain_MH13_8B_1node.json
JSON_CONFIG=${ALIGNER_PARENT_DIR}/llm_mm_aligner/experiments/aws_tranx/pretrain_MH13_70B_1node.json

PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR}:${CONDA_PYTHON_PKGS} torchrun \
    --standalone \
    --nproc_per_node=8 \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/main.py $JSON_CONFIG