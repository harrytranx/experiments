#!/bin/bash
# Usage: ./run_mast_xxs.sh pretrain_vev2.yaml vev2

# CONFIG_FILE="phase16.yaml"
# CONFIG_FILE="warmup.yaml"

CONFIG_FILE=$1
SUFFIX_NAME=$2

# CONFIG_FILE="pretrain_vev2.yaml"
NUM_NODES=1
XLFORMERS_PATH="/data/users/${USER}/fbsource/fbcode/assistant/multimodal/xlformers_llama4"
# XLFORMERS_PATH="/home/tranx/xlformers"
# MAST_ARGS=" --checkpoint_root tranx --dump_dir_id tranx_pretrain_xxs " # resharded, no optim state

CUR_DIR=$(dirname "$(realpath "$0")")
CONFIG_NAME="${CONFIG_FILE//./_}"
# RUN_NAME="${USER}-${CONFIG_NAME}-${NUM_NODES}"
RUN_NAME="${USER}-pretrain-${NUM_NODES}-${SUFFIX_NAME}"

CONFIG_FILE_PATH="$CUR_DIR/$CONFIG_FILE"

ADDITIONAL_LIBRARIES=""
ENTITLEMENT="assistant_high"
CONDA_ENV="xlformers_llama4_conda:234"
REGION="gtn"
RUNNING_REGION="pci"
WS_PATH="ws://ws.ai.pci0ai/"

echo "CUR_DIR: ${CUR_DIR}"
echo "XLFORMERS_PATH: ${XLFORMERS_PATH}"
echo "ADDITIONAL_LIBRARIES: ${ADDITIONAL_LIBRARIES}"
echo "CONFIG_FILE: ${CONFIG_FILE_PATH}"
echo "RUN_NAME: ${RUN_NAME}"

# ---------------------------
MAST_ENV="PY_SPY_DURATION=600;"
MAST_ENV+="PY_SPY_TRACE_DURATION_S=600;"
MAST_ENV+="STRUCTURED_LOGGING=1;"

MAST_ENV+="TORCH_IN_CONTAINER_RESTART=1,"
MAST_ENV+="TORCH_SKIP_STORE_BARRIER=1,"
MAST_ENV+="TORCH_ELASTIC_WORKER_IDENTICAL=1,"
MAST_ENV+="MAST_PRECHECK_FORCE_SINGLE_SERVER_RDMA_CHECK=1,"
MAST_ENV+="SKIP_CORES_TO_FAST_STOP=1,"
MAST_ENV+="AIRSTORE_LOG_TO_SECONDARY_TABLE=true,"
MAST_ENV+="ENABLE_OILFS_NUMA_BINDING=1,"
MAST_ENV+="AIRSTORE_FBPKG_ID=ws_airstore.client:prod,"
MAST_ENV+="STORAGE_FBPKG_OVERRIDE=oil.oilfs:prod,"

# NOTE(egl): v2 explodes, switch over once fixed
MAST_ENV+="HTTP_SERVER_VERSION=1;"

# MAST_ENV+="AIRSTORE_FBPKG_ID=ws_airstore.client:51442cbf545ceb67b802076c50919b86;"
# MAST_ENV+="STORAGE_FBPKG_OVERRIDE=oil.oilfs:734f6e08e1240aa0113645118f25709b;"
# MAST_ENV+="WANDB_TEAM=${WANDB_TEAM};"
# MAST_ENV+="WANDB_PROJECT=${WANDB_PROJECT};"


MAST_ENV+="FUSE_SRC=${WS_PATH}genai_fair_llm;"
MAST_ENV+="AIRSTORE_URI=${WS_PATH}airstore;"
MAST_ENV+="XLFORMERS_CLUSTER_REGION_OVERRIDE=${RUNNING_REGION}"

# ---------------------------
# MAST_ARGS=" --checkpoint_root tranx/flash_17b --dump_dir_id final_ckpt " # doesn't work
# MAST_ARGS=" --checkpoint_root tranx --dump_dir_id warmup_output_sl " # doesn't work

# MAST_ARGS=" --checkpoint_root tranx/outputs --dump_dir_id warmup_n16 " # working for training from scratch
# MAST_ARGS=" --checkpoint_root tranx/outputs --dump_dir_id warmup_n16_ve0p2 " # working for training from scratch


# MAST_ARGS=" --checkpoint_root tranx --dump_dir_id warmup_output_hyperloop " # dim mistmatch
# MAST_ARGS=" --checkpoint_root tranx --dump_dir_id warmup_output_ws128_cp1_dp32 " # resharded, no optim state

# ---------------------------
cd ${XLFORMERS_PATH}
MAST_PY_PATH="tools/launching/torchx/mast.py"

LD_LIBRARY_PATH="" torchx run \
    --scheduler_args "perpetualRun=True,conda_fbpkg_id=${CONDA_ENV},hpcClusterUuid=MastProdCluster,rmAttribution=${ENTITLEMENT},localityConstraints=region;${REGION},hpcIdentity=genai_llm_research-llama" \
    ${MAST_PY_PATH}:train \
    --sweep=${CONFIG_FILE_PATH} \
    --nodes=${NUM_NODES} \
    --name=${RUN_NAME} \
    --additional_libraries=${ADDITIONAL_LIBRARIES} \
    --h="gtt_any" \
    --enable_zswap=false \
    $MAST_ARGS \
    --env=${MAST_ENV}
