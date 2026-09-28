import argparse
import os
if os.environ.get("ACCORD_ANOMALY") == "1":  # accord: diagnostic, raise at the first NaN-producing backward op
    import torch
    torch.autograd.set_detect_anomaly(True)
if os.environ.get("ACCORD_DETERMINISTIC") == "1":
    # accord: opt-in bitwise-reproducible mode for the read-only check (set before CUDA starts)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
import socket
from scheduler import Scheduler
import torch
import subprocess

# b_default = "./configs/iid_clients.py"
b_default = "./configs/non_iid_clients.py"
parser = argparse.ArgumentParser(description='Run collaborative learning experiments')
parser.add_argument('-b', nargs='?', default=b_default, type=str,
                    help='filepath for benchmark config, default: {}'.format(b_default))
parser.add_argument('-seed', nargs='?', default=4, type=int)
args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args.b, args.seed)
scheduler.initialize()
scheduler.run_job()