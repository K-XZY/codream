#!/bin/bash
#SBATCH --job-name=codream-m2
#SBATCH --nodes=1 --gpus=1 --time=01:30:00
#SBATCH --output=/projects/u6wu/kevin/runs/codream/m2_%j.log
# M2: same smoke as the 5090 (IID, all arms, 3 epochs), then a 10-epoch CoDream-fast run alone,
# then 2 and 3 of them concurrently on the one GPU, to choose the packing.
A=$PROJECTDIR/kevin
source $A/miniforge3/etc/profile.d/conda.sh && conda activate $A/miniforge3/envs/codream
export TMPDIR=$A/tmp ACCORD_DATA=$A/data/cifar10 OMPI_MCA_btl=self,vader,tcp
cd $A/codream
R=$A/runs/codream/m2_$SLURM_JOB_ID
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 5 > $R.gpu.txt 2>/dev/null &
SMI=$!
mkdir -p $R
export ACCORD_MIN=3 ACCORD_MAX=3
for arm in cent fedavg fast; do ACCORD_DUMP=$R/smoke/ accord/run_cell.sh iid $arm 4; done
export ACCORD_MIN=10 ACCORD_MAX=10
t=$(date +%s); ACCORD_DUMP=$R/pack1/ accord/run_cell.sh iid fast 4; echo "pack1 $(( $(date +%s)-t ))s" >> $R/timing.txt
for n in 2 3; do
  t=$(date +%s)
  for s in $(seq 1 $n); do ACCORD_DUMP=$R/pack${n}_$s/ accord/run_cell.sh iid fast $((10+s)) & done; wait
  echo "pack$n $(( $(date +%s)-t ))s" >> $R/timing.txt
done
kill $SMI
echo M2_DONE >> $R/timing.txt
