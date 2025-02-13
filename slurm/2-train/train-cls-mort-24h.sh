#!/bin/bash -l

#SBATCH --array=0-4
#SBATCH --output=logs/log-train-cls-mort-24h-test-%a-%A.out
#SBATCH --partition=gpuq
#SBATCH --nodelist=cri22cn[403-405]
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100g
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4

conda activate trauma

TEST_SPLIT=$SLURM_ARRAY_TASK_ID
if [[ "$TEST_SPLIT" -eq 4 ]]; then
  VAL_SPLIT=0
else
  VAL_SPLIT=$((TEST_SPLIT + 1))
fi

srun python src/train.py fit \
--config configs/classifier.yaml \
--model.n_classes 2 \
--model.cls_target hospital_mortality \
--data.window 24 \
--data.n_splits 5 \
--data.test_split $TEST_SPLIT \
--data.val_split $VAL_SPLIT \
--trainer.logger.name "cls-mort-24h" \
--trainer.logger.version "cls-mort-24h-test-$TEST_SPLIT"

