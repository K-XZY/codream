#!/bin/bash
# Run one cell: accord/run_cell.sh SPLIT ARM SEED [GPU]
# Env: ACCORD_DUMP (run root), ACCORD_DATA (CIFAR10 root), ACCORD_MIN/MAX/PATIENCE/LOG as in configs/accord/grid.py.
# `python -m mpi4py` aborts every rank when one raises, so a crash cannot leave the other ranks waiting.
set -u
export ACCORD_SPLIT=$1 ACCORD_ARM=$2 ACCORD_GPU=${4:-0}
SEED=$3
NP=5; [ "$ACCORD_ARM" = cent ] && NP=1
cd "$(dirname "$0")/.."
LOG=${ACCORD_DUMP:-./expt_dump/accord/}/logs_${ACCORD_SPLIT}_${ACCORD_ARM}_seed${SEED}.txt
mkdir -p "$(dirname "$LOG")"
start=$(date +%s)
mpirun --oversubscribe -np $NP python -m mpi4py main.py -b ./configs/accord/grid.py -seed $SEED > "$LOG" 2>&1
rc=$?
echo "${ACCORD_SPLIT} ${ACCORD_ARM} seed${SEED} rc=$rc secs=$(( $(date +%s) - start ))" >> "$(dirname "$LOG")/cells_done.txt"
exit $rc
