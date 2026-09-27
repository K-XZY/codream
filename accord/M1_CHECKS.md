# M1 checks (RTX 5090, 2026-09-27)

Code at `8e105d0`..`accord M1: opt-in deterministic mode`.

| # | check | result |
|---|---|---|
| 1 | splits (`tests/check_runs.py` on the 9 smoke runs, seed 4): 1000 per client, no image on two clients, none in the eval set, eval 500/class, disjoint classes = {0,1},{2,3},{4,5,6},{7,8,9}, class counts match the records, the three arms of one split use the same client data, CENT data = union of the four clients | ALL CHECKS PASSED |
| 2 | diff against `393fe6b` touches only the functions in `ACCORD_CHANGES.md` | yes (listed there) |
| 3 | logging hooks are read-only: CoDream-fast IID, 5 epochs, seed 4 | see below |
| 4 | `ckpt/server_best.pt` re-scores to the logged eval/test top-1/top-3 (`tests/check_artifacts.py`; CoDream-fast IID and disjoint, FedAvg Dir(0.1), CENT disjoint) | exact |
| 5 | dream log: saved aggregate mean logits = mean of the four clients' saved logits; soft label = its softmax; shapes match | every logged epoch |
| 6 | all 9 configs, 3 epochs | rc=0 (CENT 11 s, FedAvg 19 s, CoDream-fast 94 s) |

**Check 3.** CoDream-fast is not bitwise reproducible run to run in the default mode: two runs
with logging off differ from the first dream round on (warmup is identical). With
`ACCORD_DETERMINISTIC=1` (cuDNN deterministic + deterministic algorithms, no warnings raised):

| pair | per-round client test accuracies |
|---|---|
| logging off vs logging off | identical |
| logging on vs logging on | identical |
| logging off vs logging on | identical |

So the logging does not change training. The grid runs in the default mode, as upstream does;
two runs of one cell with the same seed will therefore not match exactly for CoDream-fast
(FedAvg and CENT do, M0).

Found on the way: an exception in one MPI rank left the other ranks waiting forever;
`accord/run_cell.sh` now launches with `python -m mpi4py`, which aborts all ranks.
