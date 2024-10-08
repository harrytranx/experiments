#!/bin/bash
# Usage: nohup ./backfill.sh [LAUNCH SCRIPT] [JOB_ID] &
# Example: nohup ./backfill.sh fbcode_pretrain_MM9_70B_sbatch.sh 9322  &

# To stop:
# ps aux | grep "backfill.sh"
# kill <PID>

# nohup ./backfill.sh prod_pretrain_MM9_70B_Llama3.1_336px.sh 9299 &
# nohup ./backfill.sh prod_pretrain_MM9_70B_MH19_336px.sh 8351 &
# nohup ./backfill.sh /fsx_0/user/tranx/llm_mm_aligner/experiments/aws_tranx/mm9_stage2/prod_stage2_MM9_70B_MH19_336px.sh 15295 &

SCRIPT=$1
JOB_ID=$2

script_file=$(basename $SCRIPT)
LOG_FILE="loop_backfill_${script_file}_${JOB_ID}.log"

echo "[$(date)] Starting loop to backfill job $JOB_ID using script: $SCRIPT" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"

while true
do 
    state=$(sacct -j $JOB_ID --format=State --noheader | head -n 1 | tr -d '[:space:]')
    elapsed_time=$(sacct -j $JOB_ID --format=Elapsed --noheader | head -n 1 | tr -d '[:space:]')
    echo "[$(date)] Job = $JOB_ID, State = $state, Elapsed = $elapsed_time"

    # check for pending state
    if [ "$state" == "RUNNING" ] || [ "$state" == "PENDING" ] || [ -z "$state" ]; then
        echo "OK" 
    else 
        echo "[$(date)] Job = $JOB_ID, State = $state, Elapsed = $elapsed_time" | tee -a $LOG_FILE

        JOB_ID=$(sbatch --parsable $SCRIPT)
        echo "[$(date)] Started backfill job: $JOB_ID" | tee -a $LOG_FILE
    fi

    sleep 200 # wait a little long in case slurm auto-requeue is restarting the job already
done