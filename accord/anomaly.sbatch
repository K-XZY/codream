#!/bin/bash
#SBATCH --job-name=codream-anomaly
#SBATCH --nodes=1 --gpus=1 --time=02:00:00
#SBATCH --output=/projects/u6wu/kevin/runs/codream/anomaly_%j.log
# Guarded disjoint seed 6 (failed from epoch 16 on Isambard), adversarial weight 1.33 vs 0, 30 epochs, traced.
A=$PROJECTDIR/kevin
source $A/miniforge3/etc/profile.d/conda.sh && conda activate $A/miniforge3/envs/codream
export TMPDIR=$A/tmp ACCORD_DATA=$A/data/cifar10 ACCORD_NAN_GUARD=1 ACCORD_ANOMALY=1 ACCORD_MIN=20 ACCORD_MAX=20
cd $A/codream
ACCORD_ADV=1.33 ACCORD_DUMP=$A/runs/codream/anomaly_$SLURM_JOB_ID/adv1.33/ accord/run_cell.sh disjoint fast 6 0 &
p1=$!
wait $p1
echo ANOMALY_DONE
