#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --output=/fsx_0/user/tranx/slurm_logs/output_%j.txt
#SBATCH --error=/fsx_0/user/tranx/slurm_logs/output_%j.txt
#SBATCH --time=24:00:00
#SBATCH --account=ar-ai-hipri
#SBATCH --qos=ar-ai-hipri

ALIGNER_PARENT_DIR=$1
JSON_CONFIG=$2
CHECKPOINT_WATCH_PATH=$3
BENCHMARK_NAME=$4
CHECKPOINT_ID=$5
EVAL_PLAN=$6 # default = evals

echo "EVAL_PLAN: ${EVAL_PLAN}"
echo "JSON_CONFIG: ${JSON_CONFIG}"
echo "CHECKPOINT_WATCH_PATH: ${CHECKPOINT_WATCH_PATH}"
echo "BENCHMARK_NAME: ${BENCHMARK_NAME}"
echo "CHECKPOINT_ID: ${CHECKPOINT_ID}"
echo "EVAL_PLAN: ${EVAL_PLAN}"

# Activate conda environment
CONDA_ENV=aligner_v7
eval "$(conda shell.bash hook)"
conda activate /opt/hpcaas/.mounts/fs-036153e63d56f4dc2/home/ahmadyan/.conda/envs/aligner_v7 
echo Using conda environment: $CONDA_DEFAULT_ENV
echo CONDA_PREFIX: ${CONDA_PREFIX}

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
export PIP_NO_DEPS=true

export CUDA_DEVICE_MAX_CONNECTIONS=1


ALIGNER_DEP_DIR=$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated
CONDA_PYTHON_PKGS=${CONDA_PREFIX}/python-packages
head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
echo Node IP: $head_node_ip

CHECKPOINT_PATH=${CHECKPOINT_WATCH_PATH}/checkpoint-${CHECKPOINT_ID}
TENSORBOARD_PATH=${CHECKPOINT_WATCH_PATH}/tensorboard
mkdir -p $TENSORBOARD_PATH
echo "Running ${BENCHMARK_NAME} on $CHECKPOINT_PATH"

OUTPUT_DIR=${CHECKPOINT_WATCH_PATH}/$EVAL_PLAN/eval_results_checkpoint-${CHECKPOINT_ID}
mkdir -p $OUTPUT_DIR
USE_JSON_CONFIG=${OUTPUT_DIR}/${BENCHMARK_NAME}_${CHECKPOINT_ID}_eval_config.json
RESULT=${OUTPUT_DIR}/${BENCHMARK_NAME}_${CHECKPOINT_ID}_eval_results.txt

sed \
-e "s|CHECKPOINT_PATH|${CHECKPOINT_PATH}|g" \
-e "s|LOGGING_DIR|${TENSORBOARD_PATH}|g" \
-e "s|OUTPUT_DIR|${TENSORBOARD_PATH}|g" \
-e "s|TB_LOGDIR|${TENSORBOARD_PATH}|g" \
-e "s|EVAL_CHECKPOINT|${CHECKPOINT_ID}|g" \
$JSON_CONFIG > $USE_JSON_CONFIG

cat $USE_JSON_CONFIG

PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR}:${CONDA_PYTHON_PKGS} srun --cpus-per-gpu 24 torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    --rdzv_conf join_timeout=600,read_timeout=600 \
    ${ALIGNER_PARENT_DIR}/llm_mm_aligner/evaluate.py $USE_JSON_CONFIG $RESULT

echo "Tensorboard at ${TENSORBOARD_PATH}"

echo "DONE"



