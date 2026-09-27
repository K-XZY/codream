#!/bin/bash
#SBATCH --job-name=codream-guard
#SBATCH --nodes=1 --gpus=4
#SBATCH --output=/projects/u6wu/kevin/runs/codream/grid_guard_%j.log
# usage: sbatch --time=HH:MM:SS accord/grid.sbatch "split:arm:seed split:arm:seed ..."
# Cell i runs in slot i, on GPU i % 4; slots on one GPU run concurrently (2 per GPU measured in M2),
# cells i and i+8 run one after the other in the same slot.
A=$PROJECTDIR/kevin
source $A/miniforge3/etc/profile.d/conda.sh && conda activate $A/miniforge3/envs/codream
export TMPDIR=$A/tmp ACCORD_DATA=$A/data/cifar10 ACCORD_DUMP=$A/runs/codream/grid_guard/ ACCORD_NAN_GUARD=1
cd $A/codream
echo "commit $(git rev-parse --short HEAD) job $SLURM_JOB_ID cells: $1"
read -ra CELLS <<< "$1"
pids=()
for slot in 0 1 2 3 4 5 6 7; do
  (
    for ((i=slot; i<${#CELLS[@]}; i+=8)); do
      IFS=: read -r split arm seed <<< "${CELLS[$i]}"
      accord/run_cell.sh "$split" "$arm" "$seed" $((slot % 4))
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo GRID_DONE
