"""The 9 cells of the accord grid: split x arm, over the upstream CIFAR10 configs.

Environment variables select and adjust a run:
  ACCORD_SPLIT  iid | dir01 | disjoint
  ACCORD_ARM    cent | fedavg | fast
  ACCORD_GPU    GPU index for every MPI rank (default 0)
  ACCORD_DUMP   dump directory (default ./expt_dump/accord/)
  ACCORD_MIN, ACCORD_MAX, ACCORD_PATIENCE   stopping rule (defaults 400, 800, 20)
  ACCORD_LOG    1 | 0  accord eval, checkpoints and logs on/off (default 1)
Upstream values kept: CIFAR10 IID configs for every split (the non-IID feddream_fast config
upstream is the MNIST one; its only other difference is warmup 50 instead of 20).
"""
import os
from configs.iid_clients import fl, feddream_fast, centralized

SPLIT = os.environ.get("ACCORD_SPLIT", "iid")
ARM = os.environ.get("ACCORD_ARM", "fedavg")
GPU = int(os.environ.get("ACCORD_GPU", 0))

PREFIX = {"iid": "iid_clients", "dir01": "non_iid_balanced_clients", "disjoint": "non_iid_disjoint"}[SPLIT]
BASE = {"fedavg": fl, "fast": feddream_fast, "cent": centralized}[ARM]
ALGO = {"fedavg": "fl", "fast": "feddream_fast", "cent": "centralized"}[ARM]

cfg = dict(BASE)
cfg.update({
    "exp_type": f"{PREFIX}_{ALGO}",
    "exp_id": f"{SPLIT}-{ARM}",
    "exp_keys": [],
    "dset": "cifar10",
    "dpath": os.environ.get("ACCORD_DATA", "./imgs/cifar10"),
    "dump_dir": os.environ.get("ACCORD_DUMP", "./expt_dump/accord/"),
    "num_clients": 4,
    "samples_per_client": 1000,
    "log_tb_freq": 1,
    "accord": {"eval_holdout": 5000, "eval_seed": 0,
               "min_epochs": int(os.environ.get("ACCORD_MIN", 400)),
               "patience": int(os.environ.get("ACCORD_PATIENCE", 20)),
               "max_epochs": int(os.environ.get("ACCORD_MAX", 800)),
               "log": os.environ.get("ACCORD_LOG", "1") == "1",
               "dream_every": int(os.environ.get("ACCORD_DREAM_EVERY", 10)), "dream_slots": 8,
               "nan_guard": os.environ.get("ACCORD_NAN_GUARD", "0") == "1",
               "stable_kl": os.environ.get("ACCORD_STABLE_KL", "0") == "1"},
})
if os.environ.get("ACCORD_ADV") is not None:     # diagnostic only: weight of the adversarial term
    cfg["adv"] = float(os.environ["ACCORD_ADV"])
if SPLIT == "dir01":
    cfg["alpha"] = 0.1
if ARM == "cent":
    cfg["device_ids"] = {"node_0": [GPU]}
    if SPLIT == "iid":
        cfg["samples_per_client"] = 4000            # upstream IID CENT: one unit of 4000
    else:
        cfg["client_data_units"] = 4                # upstream non-IID CENT: union of 4 x 1000
else:
    cfg["device_ids"] = {f"node_{i}": [GPU] for i in range(5)}
current_config = cfg
