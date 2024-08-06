#!/bin/bash
#SBATCH --job-name=mh8b_retrain
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
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
head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
echo Node IP: $head_node_ip

JSON_CONFIG=${ALIGNER_PARENT_DIR}/llm_mm_aligner/experiments/aws_tranx/pretrain_MH_8B.json
# JSON_CONFIG=${ALIGNER_PARENT_DIR}/llm_mm_aligner/experiments/aws_tranx/pretrain_MH_8B_resume.json

pre_job() {
    instance_id=$(wget -q -O - http://169.254.169.254/latest/meta-data/instance-id)
    echo "PREJOB: "$(date +"%Y-%m-%d %H:%M:%S") "Running on host $(hostname), instance_id: $instance_id"
}
export -f pre_job 

# PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR} srun --cpus-per-gpu 12 bash -c 'pre_job' && torchrun \
PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR} srun --cpus-per-gpu 12 torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/main.py $JSON_CONFIG

echo "DONE"