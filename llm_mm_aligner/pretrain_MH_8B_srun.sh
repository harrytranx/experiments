#!/bin/bash
#SBATCH --job-name=mh8b_srun
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --output=/fsx_0/user/tranx/output/slurm_logs/output_%j.txt
#SBATCH --error=/fsx_0/user/tranx/output/slurm_logs/output_%j.txt
#SBATCH --time=168:00:00
#SBATCH --account=ar-ai-hipri
#SBATCH --qos=ar-ai-hipri
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive


# Activate conda environment
env_name=aligner_v5
eval "$(conda shell.bash hook)"
conda activate $env_name 
echo Using conda environment: $CONDA_DEFAULT_ENV
if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
  echo "Error: CONDA_DEFAULT_ENV is not set to $env_name"
  exit 1
fi

if [ -d "/scratch/slurm_tmpdir/$SLURM_JOB_ID" ]; then
 export TMPDIR="/fsx_0/user/{$USER}/scratch/slurm_tmpdir/$SLURM_JOB_ID"
fi

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

# launcher
ALIGNER_PARENT_DIR=/fsx_0/user/tranx
ALIGNER_DEP_DIR=/fsx_0/user/tranx/llm_mm_aligner/replicated

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=29500

PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR} srun -n $((${SLURM_JOB_NUM_NODES} * 8)) \
    python3 -u ${ALIGNER_PARENT_DIR}/llm_mm_aligner/main.py \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/experiments/aws_tranx/pretrain_MH_8B_resume.json

echo "DONE"