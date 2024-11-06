#!/bin/bash

#SBATCH --job-name=array
#SBATCH --array=1-20
#SBATCH --time=01:00:00
#SBATCH --account=midpri
#SBATCH --qos=midpri
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=1G
#SBATCH --output=output/array_%A-%a.out

# Print the task id.
srun bash -c "sleep 10; echo 'My SLURM_ARRAY_TASK_ID:' $SLURM_ARRAY_TASK_ID"