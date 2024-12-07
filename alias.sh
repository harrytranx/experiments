#!/bin/bash
alias ll='ls -l'

# count number of nodes
alias scount='sinfo -h -o %D'

# watch squeue
# alias swatch='watch -n 1 squeue --me'
alias swatch='watch -n 1 squeue -u tranx,zhenq,ahmadyan'

# get list of hosts
alias shosts="sinfo -hN|awk '{print $1}'"
sq() {
    squeue --format="%a %.18i %.9P %.10j %.15u %.2t %.10M %.6D %R"
}


alias cd_work="cd /fsx_0/user/$USER"
alias cd_exp="cd /fsx_0/user/$USER/experiments"
alias cd_fbcode="cd /fsx_0/user/$USER/fbcode"
alias cd_rsync="cd /fsx_0/user/tranx/rsync/llm_mm_aligner"
alias cd_aws="cd /fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws"
alias cd_clip="cd /fsx_0/user/tranx/github/openCLIPMeta"
alias jpt="jupyter-lab --ip=0.0.0.0 --port=8921 --no-browser > ~/logs/jupyter-lab.log 2>&1 &"
alias jpt_kill="pkill -f jupyter"

kill_grep() {
    pattern=$1
    pgrep -f $pattern | xargs sudo kill
}

kill_zombie() {
    zombie_pids=$(ps -eo pid,stat | awk '$2 ~ /Z/ {print $1}')
    # Check if there are any zombie processes
    if [ -z "$zombie_pids" ]; then
        echo "No zombie processes found."
        exit 0
    fi
    echo "Found zombie processes with PIDs: $zombie_pids"
    # Iterate over each zombie PID
    for pid in $zombie_pids; do
        # Get the parent process ID (PPID) of the zombie process
        ppid=$(ps -o ppid= -p $pid)
        # Check if PPID is valid
        if [ -n "$ppid" ]; then
            echo "Killing parent process with PID: $ppid"
            # Attempt to kill the parent process
            sudo kill -9 $ppid
        else
            echo "Could not find parent process for zombie PID: $pid"
        fi
    done
    echo "Attempted to kill all parent processes of zombie processes."
}

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
    srun --account=midpri --qos=midpri -N 1 -n 1 --cpus-per-task 24 --gpus-per-task=8 --job-name=dev --mem=32000 --pty /bin/bash -ls
}

sbash_cpu() {
    srun --account=ar-ai-hipri --partition=cpu -N 1 -n 1 --cpus-per-task 48 --job-name=bash --mem=32000 --pty /bin/bash
    # srun --account=ar-ai-hipri --partition=cpu -N 2 -n 2 --cpus-per-task 48 --job-name=bash --mem=32000 --pty /bin/bash
}

sbash_midpri() {
    srun --account=midpri --qos=midpri -N 1 -n 1 --cpus-per-task 24 --gpus-per-task=8 --job-name=dev --mem=32000 --pty /bin/bash -ls
}

sbash_q2() {
    srun --account=ar-ai-hipri --partition=q2 -N 1 -n 1 --cpus-per-task 24 --gpus-per-task=8 --job-name=dev --mem=32000 --pty /bin/bash -ls
}

slast() {
    last_job_id=$(sacct -u tranx -X --start now-3hours -o jobid | tail -n 1 | xargs)
    echo "$last_job_id"
}

slog() {
    if [ -n "$1" ]; then
        job_id=$1
    else 
        job_id=$(slast)
    fi
    output_file=$(wlog $job_id)   
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

get_log(){
    # scontrol will lose information very quickly, better use sacct
    job_id=$1
    # log_file=$(scontrol show job $job_id | grep StdOut | awk '{print $1}' | cut -d'=' -f2)
    log_file=$(sacct -j $job_id --format=JobID,StdOut%500 | awk 'NR==3 {print $2}')

    username=$(slurm_get_user $job_id)

    log_file=${log_file/\%u/$username}
    log_file=${log_file/\%j/$job_id}

    echo "$log_file"
}

wlog() {
    if [ -n "$1" ]; then
        job_id=$1
        # log_file="/fsx_0/user/tranx/slurm_logs/output_$job_id.txt"
    else 
        job_id=$(slast)
    fi 

    # log_file="/fsx_0/user/tranx/slurm_logs/output_$job_id.txt"
    log_file=$(get_log $job_id)
    echo "$log_file"
}

slurm_get_user() {
    job_id=$1
    user=$(squeue -j $job_id -o "%u" | awk 'NR>1 {print $0}')
    echo "$user"
}


jrun() {
    sbatch /fsx_0/user/tranx/experiments/sbatch_jupyter_lab.sh
    tail -f /data/home/tranx/logs/jupyter_lab.log
}

jkernel() {
    cat ~/logs/jupyter_lab.log | grep "h100-st-p548xlarge"
}


sgrep() {
    if [ -n "$1" ]; then
        job_id=$1
    else 
        job_id=$(slast)
    fi
    
    output_file=$(wlog $job_id) 
    sgrep_f $output_file
}

sgrep_f() {
    file=$1
    grep -e "error" \
    -e "out of memory" \
    -e "permission" \
    -e "'loss':" \
    -e "/perception_tokenizer.pt" \
    -e "Training completed" \
    -e "CANCELLED" \
    -e "uncorrectable ECC" \
    -i $file
}

shold() {
    job_id=$1 
    echo "Holding job $job_id into queue"
    scontrol requeuehold $job_id 
}

srelease() {
    job_id=$1 
    echo "Releasing job $job_id"
    scontrol release $job_id 
}


sjob(){
    # show useful information about a specific job
    job_id=$1 
    echo $(scontrol show job $job_id | grep -E "Command")
    echo $(scontrol show job $job_id | grep -E "WorkDir")
    echo $(scontrol show job $job_id | grep -E "StdErr")
    echo $(scontrol show job $job_id | grep -E "StdOut")
}


sutil()
{   
    job_id=$1
    python /fsx_0/user/ahmadyan/utilization.py $job_id
}


# TMUX
tlist() {
    tmux list-sessions
}

tin() {
    name=$1
    tmux attach -t $name
}

tout() {
    tmux detach
}

tnew() {
    name=$1
    tmux new-session -t $name
}

alias s2="PYTHONPATH=PYTHONPATH:/fsx_0/user/tranx/experiments python /fsx_0/user/tranx/experiments/lib/stool.py"