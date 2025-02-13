#!/bin/bash -l

#SBATCH --array=0-4
#SBATCH --output=logs/log-train-cls-iss-48h-test-%a-%A.out
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
--model.n_classes 3 \
--model.cls_target iss_tercile \
--data.window 48 \
--data.n_splits 5 \
--data.test_split $TEST_SPLIT \
--data.val_split $VAL_SPLIT \
--trainer.logger.name "cls-iss-48h" \
--trainer.logger.version "cls-iss-48h-test-$TEST_SPLIT"

