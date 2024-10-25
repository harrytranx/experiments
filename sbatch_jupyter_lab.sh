#!/bin/bash
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=32000
#SBATCH --output=/data/home/tranx/logs/jupyter_lab.log
#SBATCH --error=/data/home/tranx/logs/jupyter_lab.log
#SBATCH --account=midpri
#SBATCH --qos=midpri

eval "$(conda shell.bash hook)"
conda activate aws
echo Using conda environment: $CONDA_DEFAULT_ENV

jupyter-lab --ip=0.0.0.0 --port=8921 --no-browser