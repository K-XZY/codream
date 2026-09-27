# Environment (M0, 2026-09-27)

**RTX 5090 box (`ubuntu`)**: conda env `codream` at `/mnt/data/Work/envs/codream`, Python 3.10,
torch 2.11.0+cu128, torchvision 0.26.0+cu128 (capability (12, 0)), Open MPI 5.0.11 + mpi4py
(conda-forge), numpy, jmespath, tensorboardX, matplotlib, pillow, medmnist. `requirements.txt`
is not used (2022 pins, no sm_120 torch).

Launch (from the repo root; rank 0 = server, ranks 1-4 = clients):

    mpirun --oversubscribe -np 5 python main.py -b ./configs/accord/grid.py -seed 4   # FedAvg, CoDream-fast
    mpirun --oversubscribe -np 1 python main.py -b ./configs/accord/grid.py -seed 4   # CENT

## M0 results (upstream code, IID, 2 epochs, all ranks on GPU 0)

| arm | exit | wall clock | peak GPU memory |
|---|---|---|---|
| CENT | 0 | 1082 s (almost all the one-time CIFAR10 download) | 2.6 GB |
| FedAvg | 0 | 13 s | 11.4 GB (5 processes) |
| CoDream-fast (`warmup` 1) | 0 | 70 s | 18.9 GB (5 processes) |

- Each client's class fractions sum to 1 (1000 images) in every run.
- Same seed twice (FedAvg): identical client splits and identical test accuracy per round
  (0.1052, 0.1013), so M1 check 3 is exact.
- Seconds per epoch at full length are measured in M1 and M2 (2-epoch runs are dominated by start-up).
