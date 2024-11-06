#!/bin/bash

#SBATCH --job-name=array
#SBATCH --time=01:00:00
#SBATCH --account=midpri
#SBATCH --qos=midpri
#SBATCH --ntasks-per-node=2
#SBATCH --mem=1G
#SBATCH --output=output/output_%j_%N.out
#SBATCH --error=output/output_%j_%N.out
#SBATCH --nodes=2

# Print the task id.
srun bash -c "sleep 10; echo 'My SLURM_NODEID:' $SLURM_NODEID"