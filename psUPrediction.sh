#!/bin/bash

# Settings to be modified:

# gres setup:
# - gpu:tesla is currently fixed as we only have GPUs as extract resource
# - Moreover we only have cards of the tesla type
# - The number after the clon specifies the number of GPUs required,
# e.g. something between 1 and 4


# Modifiy other SLURM variables as needed

#SBATCH --job-name="PRD_3"
#SBATCH --output=MLP_3.txt
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=50G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=lemsaraamina@gmail.com
#SBATCH -p gpu
#SBATCH --gres=gpu:turing:1

srun /home/alemsara/miniconda3/envs/DirectRNA/bin/python /prj/Isabel_ONT_rRNA/scripts/MLP_Y.py



# Template for SLURM GPU handling scripts
# From https://techfak.net/gpu-cluster

# please do not modify the following parts

