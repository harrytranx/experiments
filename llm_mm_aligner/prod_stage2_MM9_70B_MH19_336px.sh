#!/bin/bash
#SBATCH --job-name=stage2
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --output=/fsx_0/user/tranx/output/slurm_logs/output_%j.txt
#SBATCH --error=/fsx_0/user/tranx/output/slurm_logs/output_%j.txt
#SBATCH --time=168:00:00
#SBATCH --account=ar-ai-hipri
#SBATCH --qos=ar-ai-hipri
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
#SBATCH --exclude=h100-st-p548xlarge-4

# Activate conda environment
CONDA_ENV=aligner_v7
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV 
echo Using conda environment: $CONDA_DEFAULT_ENV
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
  echo "Error: CONDA_DEFAULT_ENV is not set to $CONDA_ENV"
  exit 1
fi

echo Node list: $SLURM_JOB_NODELIST


# cluster tunning
export LOGLEVEL=INFO
export CUDA_LAUNCH_BLOCKING=0
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_BUFFSIZE=8388608 
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="enp"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker0,nerdctl0,veth*"
export NCCL_IB_DISABLE=1
export CUDA_CACHE_PATH="/fsx_0/user/${USER}/.nv/ComputeCache"
export LD_PRELOAD=/usr/local/cuda-12.3/lib/libnccl.so
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# launcher
ALIGNER_PARENT_DIR=/fsx_0/user/tranx
ALIGNER_DEP_DIR=$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated
CONDA_PYTHON_PKGS=${CONDA_PREFIX}/python-packages
head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
echo Node IP: $head_node_ip

MM9_CONF_DIR="/fsx_0/user/tranx/experiments/llm_mm_aligner/stage2_mm9"
JSON_CONFIG=$MM9_CONF_DIR/stage2_MM9_70B_MH19_336px_64nodes.json

echo "Using config from: $JSON_CONFIG"

PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR}:${CONDA_PYTHON_PKGS} srun --cpus-per-gpu 24 torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/main.py $JSON_CONFIG

echo "DONE"