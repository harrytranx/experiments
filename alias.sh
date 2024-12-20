#!/bin/bash
alias ll='ls -l'

# count number of nodes
alias scount='sinfo -h -o %D'

# watch squeue
# alias swatch='watch -n 1 squeue --me'
alias swatch='watch -n 1 squeue -u tranx,zhenq,ahmadyan,cyprien,xuanhu'

# get list of hosts
alias shosts="sinfo -hN|awk '{print $1}'"
my_squeue() {
    squeue --format="%a %.10i %.9P %.10j %.15u %.2t %.10M %.6D %R"
}

sqlong() {
    name_length=$1
    # squeue --format="%a %.18i %.9P %.50j %.15u %.2t %.10M %.6D"
    # squeue --format="%a %.18i %.50j %.15u %.2t %.10M %.6D"
    squeue --format="%.10i %.70j %.15u %.2t %.10M %.6D"
}

sq() {
    local name_filter=""
    local user_filter=""
    local state_filter=""
    local partition_filter="q1" # default to q1 if not provided
    local count_flag=false
    local action_flag=""
    
    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --name|-n) 
                name_filter="$2"
                count_flag=true
                shift 2
                ;;
            --user|-u) 
                user_filter="$2"
                count_flag=true
                shift 2
                ;;
            --state|-s) 
                state_filter="$2"
                count_flag=true
                shift 2
                ;;
            --partition|-p) 
                partition_filter="$2"
                count_flag=true
                shift 2
                ;;
            --count|-c) 
                count_flag=true
                shift
                ;;
            --long|-l) 
                sqlong
                return 1
                ;;
            --do) 
                action_flag="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: sq [options]"
                echo ""
                echo "Options:"
                echo "  -n, --name          Filter by job name"
                echo "  -u, --user          Filter by user"
                echo "  -s, --state         Filter by state (PD, R, etc.)"
                echo "  -p, --partition     Filter by partition (q1, q2, cpu)"
                echo "  -c, --count         Count number of nodes"
                echo "  -l, --long          Print in format with longer job name"
                echo "  --do                Do action on resulting jobs (cancel, hold, release)"
                echo "  -h, --help          Print this help message"
                return 1
                ;;
            *) 
                echo "Unknown option: $1" 
                return 1 
                ;;
        esac
    done

    echo "filter based on: name=$name_filter, user=$user_filter, state=$state_filter, partition=$partition_filter"
    echo "flags: count_flag=$count_flag"

    # Filter squeue output
    local queue_output
    queue_output=$(my_squeue | awk -v name="$name_filter" -v user="$user_filter" -v state="$state_filter" -v partition="$partition_filter" -v count_flag="$count_flag" '
    BEGIN {
    }
    NR == 1 {print; next} # Print header
    {
        if ((name == "" || index($4, name) == 1) &&
            (user == "" || index($5, user) == 1) &&
            (partition == "" || index($3, partition) == 1) &&
            (state == "" || index($6, state) == 1)) {
            print
            total_nodes[$6] += $8
            user_nodes[$5][$6] += $8
        }
    }
    END {
        if (count_flag == "true") {
            # Print total nodes grouped by user and state
            printf "\n%-15s %-10s %s\n", "USER", "STATE", "TOTAL NODES"
            printf "%s\n", "-----------------------------------"
            for (u in user_nodes) {
                for (s in user_nodes[u]) {
                    printf "%-15s %-10s %d\n", u, s, user_nodes[u][s]
                }
            }
            # Print total nodes grouped by state
            printf "\n%-10s %s\n", "STATE", "TOTAL NODES"
            printf "%s\n", "----------------------"
            for (s in total_nodes) {
                printf "%-10s %d\n", s, total_nodes[s]
            }
        }
    }')

    echo "$queue_output"

    # Validate action_flag if provided
    if [ -z "$action_flag" ]; then 
        return 1 
    fi 

    if [[ "$action_flag" != "cancel" && "$action_flag" != "hold" && "$action_flag" != "release" ]]; then
        echo "Invalid action: $action_flag. Valid actions are [cancel, hold, release]."
        return 1
    fi



    echo -n "Enter pass_phrase ($action_flag): "
    read -r pass_phrase
    if [ "$pass_phrase" != "$action_flag" ]; then
        echo "Invalid pass_phrase"
        return 1
    fi

    local job_ids
    job_ids=$(echo "$queue_output" | awk 'NR > 2 && $2 ~ /^[0-9]+$/ {print $2}')
    process_job_ids "$job_ids" $action_flag

}

process_job_ids() {
    local job_ids="$1" # A space-separated list of job IDs
    local action="$2"
    # echo $action

    while read -r jobid; do
        if [ "$action" = "cancel" ]; then
            echo "Cancelling jobid=$jobid"
            scancel $jobid
        elif [ "$action" = "hold" ]; then
            echo "Holding jobid=$jobid"
            shold $jobid
        elif [ "$action" = "release" ]; then
            echo "Releasing jobid=$jobid"
            srelease $jobid
        else
            echo "Unknown action: $action for jobid=$jobid"
        fi
    done <<< "$job_ids"
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
    if [ -n "$1" ]; then
        name=$1
    else 
        name="dev"
    fi
    srun --exclusive --account=midpri --qos=midpri -N 1 -n 1 --cpus-per-task 24 --gpus-per-task=8 --job-name=$name --mem=32000 --pty /bin/bash -ls
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


get_held_jobs() {
    # Get the list of all held jobs
    held_jobs=$(squeue -h -o "%i %t" | awk '$2 == "PD" {print $1}')
    
    if [ -z "$held_jobs" ]; then
        echo "No held jobs found."
        return 1
    fi

    echo "$held_jobs"
    return 0
}

shold_old() {
    job_id=$1 
    echo "Holding job $job_id into queue"
    scontrol requeuehold $job_id 
}

shold() {
    job_id=$1
    hold_time=$2
   
    if [ -z "$job_id" ]; then
        echo "Error: Please provide a job ID."
        echo "Usage: shold <job_id>"
        return 1
    fi

    if [ -z "$hold_time" ]; then
        hold_time=10 # default 10 mins
    fi

    echo "Holding job $job_id into queue..."
    scontrol requeuehold "$job_id"
    
    if [ $? -ne 0 ]; then
        echo "Failed to hold job $job_id."
        return 1
    fi

    echo "Job $job_id is held. It will be released after $hold_time minutes..."

    # Schedule release 
    hold_time_seconds=$((60 * hold_time))
    (sleep $hold_time_seconds && scontrol release "$job_id" || echo "Failed to release job $job_id.") &
}


srelease() {
    job_id=$1 
    echo "Releasing job $job_id"
    scontrol release $job_id 
}

srelease_all() {
    # Get the list of held jobs using the first function
    held_jobs=$(get_held_jobs)
    
    if [ $? -ne 0 ]; then
        echo "No jobs to release."
        return 0
    fi

    echo "Releasing all held jobs..."
    
    for job_id in $held_jobs; do
        output=$(scontrol release "$job_id" 2>&1)
        if [ $? -eq 0 ]; then
            echo "Job $job_id released successfully."
        else
            echo "Failed to release job $job_id. Error: $output"
        fi
    done
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

# AWS S3
s3count() {
    
    local s3_path="$1"
    if [[ -z "$s3_path" ]]; then
        echo "Usage: s3count s3://ar-ai-s3-use2/dataset01/metaclip_v2_2b_090924"
        return 1
    fi

    # Use aws s3 ls to list objects and calculate size and file count
    local total_size file_count
    total_size=0
    file_count=0

    # Iterate through the lines output by aws s3 ls
    while read -r line; do
        # Extract size from the output and add to total size
        size=$(echo "$line" | awk '{print $3}')
        if [[ -n "$size" && "$size" =~ ^[0-9]+$ ]]; then
        total_size=$((total_size + size))
        file_count=$((file_count + 1))
        fi
    done < <(aws s3 ls "$s3_path" --recursive)

    local total_size_tb
    total_size_tb=$(echo "scale=4; $total_size / (1024^4)" | bc)

    echo "Total size: $total_size bytes ($total_size_tb TB)"
    echo "Number of files: $file_count"
}


sdelete() {
    local dir_to_delete=$1
    local empty_dir=/tmp/empty_dir
    # Create an empty directory under /tmp, deleting it first if it already exists
    rm -rf "$empty_dir"
    mkdir -p "$empty_dir"
    # # Show the dry-run output first
    # echo "Dry-run output:"

    # Show the dry-run summary
    echo "Dry-run summary:"
    # rsync -nav --delete --summary "$empty_dir/" "$dir_to_delete/"
    rsync -na --delete --stats "$empty_dir/" "$dir_to_delete/"

    # rsync -na --delete "$empty_dir/" "$dir_to_delete/"
    # Prompt user to enter phrase: delete
    read -p "Enter 'delete' to confirm deletion: " confirmation
    # If phrase is entered correctly, go ahead and do the delete
    if [ "$confirmation" = "delete" ]; then
        echo "Deleting files..."
        rsync -av --delete "$empty_dir/" "$dir_to_delete/"
    else
        echo "Deletion cancelled."
    fi
    # Remove the temporary empty directory
    rmdir "$empty_dir"
}

s3count_m2c2() {
    for x in {0..99}; do
        echo $x
        s3count s3://ar-ai-s3-use2/dataset01/metaclip_v2_2b_090924/$x/
    done
}

s3count_metaclip_v2() {
    for x in {0..99}; do
        echo $x
        s3count s3://ar-ai-s3-use2/dataset01/metaclip_v2/shards/$x/
    done
}