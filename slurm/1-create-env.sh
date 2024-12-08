#!/bin/bash -l

#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=2g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output="slurm-log-1-create-env-%j.out"

module load gcc/12.1.0

conda env create -f env.yml
