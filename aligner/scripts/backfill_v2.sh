#!/bin/bash
# Usage: nohup ./backfill.sh JOB_ID "RELAUNCH COMMAND" &
# Example: nohup ./backfill.sh 9322 "sbatch --parsable --job-name=s2_exp30 --nodes=160 launch_sbatch_h100s.sh mm9_stage2/stage2_MM9_70B_MH19_336px_128nodes_exp30.json" &

# To stop:
# ps aux | grep "backfill.sh"
# kill <PID>


JOB_ID=$1
RELAUNCH_CMD=$2

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

        JOB_ID=$(eval "$RELAUNCH_CMD")
        echo "[$(date)] Started backfill job: $JOB_ID" | tee -a $LOG_FILE
    fi

    sleep 200 # wait a little long in case slurm auto-requeue is restarting the job already
done