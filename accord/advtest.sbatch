#!/bin/bash
#SBATCH --job-name=codream-advtest
#SBATCH --nodes=1 --gpus=2 --time=02:00:00
#SBATCH --output=/projects/u6wu/kevin/runs/codream/advtest_%j.log
# Guarded disjoint seed 6 (failed from epoch 16 on Isambard), adversarial weight 1.33 vs 0, 30 epochs, traced.
A=$PROJECTDIR/kevin
source $A/miniforge3/etc/profile.d/conda.sh && conda activate $A/miniforge3/envs/codream
export TMPDIR=$A/tmp ACCORD_DATA=$A/data/cifar10 ACCORD_NAN_GUARD=1 ACCORD_NANDEBUG=1 ACCORD_MIN=30 ACCORD_MAX=30
cd $A/codream
ACCORD_ADV=1.33 ACCORD_DUMP=$A/runs/codream/advtest_$SLURM_JOB_ID/adv1.33/ accord/run_cell.sh disjoint fast 6 0 &
p1=$!
ACCORD_ADV=0 ACCORD_DUMP=$A/runs/codream/advtest_$SLURM_JOB_ID/adv0/ accord/run_cell.sh disjoint fast 6 1 &
p2=$!
wait $p1 $p2
echo ADVTEST_DONE
