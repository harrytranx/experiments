job_id=$1
log_file=/fsx_0/user/tranx/output/slurm_logs/output_$job_id.txt
log_file_pruned=output_${job_id}_pruned.txt

search_strings=(
  "NCCL INFO NCCL_TOPO_FILE"
  "NCCL INFO NVLS"
  "Setting worker"
  "FileTimerServer"
  "reply file to"
  "'TORCHELASTIC_ENABLE_FILE_TIMER'"
  "NCCL INFO Channel"
  "NCCL INFO Connected all trees"
  "NCCL INFO NVLS comm"
  "NCCL INFO threadThresholds"
  "p2p channels per peer"
  "threadThresholds"
)

# Use sed to edit the file and remove the lines containing any of the search strings
sed -i "/${search_strings[*]}/d" $log_file