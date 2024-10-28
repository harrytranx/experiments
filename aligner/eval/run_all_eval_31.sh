# Eval

MODEL_PATH=/fsx_0/checkpoints/tranx/MM9-Pretrain-70B/Llama31_336px_128nodes_bz32_scratch

CHECKPOINT_START=1000
CHECKPOINT_END=2000
CHECKPOINT_INTERVAL=1000

# EVAL_CONFIG_DIR="/home/ahmadyan/eval_31"
EVAL_CONFIG_DIR="/fsx_0/user/tranx/eval/llm_mm_aligner/experiments/aws_adel/eval_31"

echo "Running eval on $MODEL_PATH: checkpoint $CHECKPOINT_START-$CHECKPOINT_END"

# benchmarks="ai2d vqa mmmu chartqa docvqa infographics infographics_w_ocr mathvista mmbench textvqa "
benchmarks="ai2d vqa "
IFS=" "
for benchmark in ${benchmarks}; do
    echo "===================="
    echo "sh loop.sh ${EVAL_CONFIG_DIR}/eval_${benchmark}.json ${MODEL_PATH} $benchmark $CHECKPOINT_START $CHECKPOINT_END $CHECKPOINT_INTERVAL"
    sh loop.sh ${EVAL_CONFIG_DIR}/eval_${benchmark}.json ${MODEL_PATH} $benchmark $CHECKPOINT_START $CHECKPOINT_END $CHECKPOINT_INTERVAL
done