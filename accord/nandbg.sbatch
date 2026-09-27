#!/bin/bash
#SBATCH --job-name=codream-nandbg
#SBATCH --nodes=1 --gpus=1 --time=01:30:00
#SBATCH --output=/projects/u6wu/kevin/runs/codream/nandbg_%j.log
# Trace where CoDream-fast's dreams first go non-finite: disjoint classes, seed 6 (NaN at epoch 19 in M4).
A=$PROJECTDIR/kevin
source $A/miniforge3/etc/profile.d/conda.sh && conda activate $A/miniforge3/envs/codream
export TMPDIR=$A/tmp ACCORD_DATA=$A/data/cifar10 ACCORD_NANDEBUG=1 ACCORD_DREAM_EVERY=1 ACCORD_MIN=40 ACCORD_MAX=40
cd $A/codream
ACCORD_DUMP=$A/runs/codream/nandbg_$SLURM_JOB_ID/ accord/run_cell.sh disjoint fast 6
