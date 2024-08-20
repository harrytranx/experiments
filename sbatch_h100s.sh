#!/bin/bash
#SBATCH --job-name=sample_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --output=/fsx_0/user/%u/slurm_logs/output_%j.txt
#SBATCH --output=/fsx_0/user/%u/slurm_logs/output_%j.txt
#SBATCH --time=168:00:00
#SBATCH --account=ar-ai-hipri
#SBATCH --qos=ar-ai-hipri
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive

RUN_MODULE=$1
JSON_CONFIG=$2
CONDA_ENV_PATH=$3
ALIGNER_PARENT_DIR=$4

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_PATH
echo Using conda environment: $CONDA_DEFAULT_ENV
if [ "$CONDA_DEFAULT_ENV" != "$(basename "$CONDA_ENV_PATH")" ]; then
  echo "Error: CONDA_DEFAULT_ENV is not set to $CONDA_ENV"
  exit 1
fi

echo Node list: $SLURM_JOB_NODELIST

head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
echo Node IP: $head_node_ip

# cluster tunning - specific for H100s 
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
echo "Run module: $RUN_MODULE"
echo "Config: $JSON_CONFIG"

PYTHON_PATH_EXTRAS=$PYTHONPATH:\
$ALIGNER_PARENT_DIR:\
$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated:\
$CONDA_ENV_PATH/python-packages

PYTHONPATH=${PYTHONPATH}:${PYTHON_PATH_EXTRAS} srun --cpus-per-gpu 24 torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    ${RUN_MODULE} $JSON_CONFIG

echo "DONE"