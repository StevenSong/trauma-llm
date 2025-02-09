#!/bin/bash -l

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4

conda activate trauma

srun python src/train.py fit --config configs/classifier.yaml --model.n_classes 3 --model.cls_target iss_tercile --data.test_split 0 --data.val_split 1 --trainer.logger.version "cls-iss-test-0"

