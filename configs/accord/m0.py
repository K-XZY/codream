"""M0 smoke configs: upstream CIFAR10 IID configs, only epochs, devices and paths overridden.

Select the arm with the environment variable ACCORD_ARM in {cent, fedavg, fast}.
"""
import os
from configs.iid_clients import fl, feddream_fast, centralized

ONE_GPU5 = {f"node_{i}": [0] for i in range(5)}
COMMON = {"epochs": int(os.environ.get("ACCORD_EPOCHS", 2)),
          "dump_dir": "./expt_dump/accord_m0/", "dpath": "./imgs/cifar10"}

ARMS = {
    "fedavg": dict(fl, **COMMON, device_ids=ONE_GPU5),
    "fast": dict(feddream_fast, **COMMON, device_ids=ONE_GPU5, warmup=1),
    # upstream `centralized` points at SVHN; the paper's CENT is CIFAR10 with 4 x 1000 images
    "cent": dict(centralized, **COMMON, dset="cifar10", device_ids={"node_0": [0]}),
}
current_config = ARMS[os.environ.get("ACCORD_ARM", "fedavg")]
