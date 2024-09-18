#!/bin/bash
alias ll='ls -l'

# count number of nodes
alias scount='sinfo -h -o %D'

# watch squeue
alias swatch='watch -n 1 squeue --me'

# get list of hosts
alias shosts="sinfo -hN|awk '{print $1}'"

alias cd_work="cd /fsx_0/user/$USER"
alias cd_exp="cd /fsx_0/user/$USER/experiments"
alias cd_fbcode="cd /fsx_0/user/$USER/fbcode"
alias cd_rsync="cd /fsx_0/user/tranx/rsync/llm_mm_aligner"
alias cd_aws="cd /fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws"
alias jpt="jupyter-lab --ip=0.0.0.0 --port=8921 --no-browser > ~/logs/jupyter-lab.log 2>&1 &"
alias jpt_kill="pkill -f jupyter"

ibatch() {
    script=$1
    # json_config=$2
    job_id=$(sbatch --parsable $script)
    # job_id=$(sbatch --export=JSON_CONFIG=$json_config --parsable $script)

    echo "job_id = $job_id"
    # output_file="/fsx_0/user/$USER/output/slurm_logs/output_$job_id.txt"
    output_file=$(wlog $job_id)  
    echo "Waiting for output from: $output_file" 

    while [ ! -f $output_file ]; do
        sleep 1
    done

    tail -f $output_file
}

ssh_node() {
    node_id=$1
    ssh -A "h100-st-p548xlarge-$node_id"
}

sbash() {
    srun --account=ar-ai-hipri --qos=ar-ai-hipri -N 1 -n 1 --cpus-per-task 16 --gpus-per-task=8 --job-name=dev --mem=32000 --pty /bin/bash -ls
}

slast() {
    last_job_id=$(sacct -u tranx -X --start now-3hours -o jobid | tail -n 1 | xargs)
    echo "$last_job_id"
}

slog() {
    if [ -n "$1" ]; then
        job_id=$1        output_file=$(wlog $job_id)         last_job_id=$(slast)
        output_file=$(wlog $last_job_id)    
        while [ ! -f $output_file ]; do
            echo "Waiting for output from: $output_file" 
            sleep 5
        done
    fi 

    tail -f $output_file
}


scat(){
    if [ -n "$1" ]; then
        job_id=$1
    else 
        job_id=$(slast)
    fi
    
    output_file=$(wlog $job_id)    
    cat $output_file
}


sless() {
    if [ -n "$1" ]; then
        job_id=$1
    else 
        job_id=$(slast)
    fi
    
    output_file=$(wlog $job_id)    
    less $output_file
}

plog() {
    job_id=$1
    output_file=$(wlog $job_id) 
    echo $output_file
    cat $output_file | grep "Evaluating:"
}

wlog() {
    if [ -n "$1" ]; then
        job_id=$1
        log_file="/fsx_0/user/tranx/slurm_logs/output_$job_id.txt"
    else 
        job_id=$(slast)
    fi 

    log_file="/fsx_0/user/tranx/slurm_logs/output_$job_id.txt"
    echo "$log_file"
}

sq() {
    squeue --format="%a %.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
}

start_jpt() {
    sbatch /data/home/tranx/sbatch_jupyter_lab.sh
    tail -f /data/home/tranx/logs/jupyter_lab.log
}

which_jpt() {
    cat logs/jupyter_lab.log | grep 8921
}

sgrep() {
    if [ -n "$1" ]; then
        job_id=$1
    else 
        job_id=$(slast)
    fi
    
    output_file=$(wlog $job_id) 
    grep -e "error" \
        -e "out of memory" \
        -e "permission" \
        -e "'loss':" \
        -i $output_file
}