# Isambard (M2, 2026-09-27)

- Login `ssh i1` (project **u6wu**; u6nf rejects jobs with `AssocGrpCPUMinutesLimit`).
- Repo `$PROJECTDIR/kevin/codream` (branch `accord-repro`, pulled from a login session).
- Env `$PROJECTDIR/kevin/miniforge3/envs/codream`: Python 3.10, torch/torchvision cu126 (aarch64),
  Open MPI 5.0.11 + mpi4py (conda-forge); built on a compute node by `accord_build_env.sh`.
- CIFAR10 `$PROJECTDIR/kevin/data/cifar10` (downloaded once; ~15 min at ~200 kB/s).
- Runs `$PROJECTDIR/kevin/runs/codream/grid/`; job logs `$PROJECTDIR/kevin/runs/codream/grid_<job>.log`.

## M2 results (job 6904824, one GH200)

| check | result |
|---|---|
| IID smoke, 3 epochs, all arms | rc=0; CENT 29 s, FedAvg 32 s, CoDream-fast 90 s |
| `tests/check_runs.py` | ALL CHECKS PASSED |
| client and eval index hashes vs the 5090, seed 4 | identical (`6b379182 494a5909 4d7faf20 0515a82a`, eval `399ec1b5`) |
| epoch-1 CoDream-fast client-mean eval top-1 | 28.9 (5090: 27.3), within CoDream-fast's run-to-run spread (M1 check 3) |
| one CoDream-fast run, 10 epochs + 20 warm-up | 347 s; ~11 s for a plain round, ~31 s per round averaged over rounds 1–9 |
| two concurrent on one GPU | 614 s each (1.13× throughput), 61 GB peak |
| three concurrent | not measured (the probe script's `wait` also waited on its GPU sampler and hung; cancelled) |

**Packing:** two runs per GPU, eight per node (`accord/grid.sbatch`). Three would sit near the
95 GiB ceiling for a ~13 % gain at most.

**Projection:** CoDream-fast ~30–35 s per epoch alone, ~55 s packed, before the server's
distillation (every 5th round from round 10) is counted; 400 epochs ≈ 4–7 h, the 800 cap ≈ 8–13 h.
FedAvg and CENT are minutes to an hour. Jobs go to `workq` with a 20 h limit.
