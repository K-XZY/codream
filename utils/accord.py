"""Additions for the accord reproduction (see ACCORD_CHANGES.md).

Everything here is inactive unless the config carries an "accord" block:

    "accord": {"eval_holdout": 5000, "eval_seed": 0,
               "min_epochs": 400, "patience": 20, "max_epochs": 800,
               "log": True, "dream_every": 10, "dream_slots": 8}

Nothing in this module draws from the global numpy or torch random streams, so turning the
logging on or off does not change what training sees.
"""
import json
import os
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Disjoint classes split: clients 0-1 hold two classes each, clients 2-3 three each.
DISJOINT_CLASSES = [[0, 1], [2, 3], [4, 5, 6], [7, 8, 9]]


def enabled(config):
    return "accord" in config


def acfg(config):
    return config["accord"]


# ---------------------------------------------------------------- data
def carve_eval(dset_obj, config):
    """Remove a class-balanced eval set from the training set, in place.

    Afterwards dset_obj.train_dset holds only the remaining images, dset_obj.train_orig_idx maps
    their positions to indices in the original CIFAR10 training set, and dset_obj.eval_dset is the
    held-out set with the (deterministic) test transform.
    """
    a = acfg(config)
    n_eval = a.get("eval_holdout", 0)
    train = dset_obj.train_dset
    targets = np.array(train.targets)
    n_cls = dset_obj.NUM_CLS
    rng = np.random.RandomState(a.get("eval_seed", 0))  # private stream
    eval_idx = []
    for c in range(n_cls):
        idx_c = np.where(targets == c)[0]
        eval_idx.extend(rng.choice(idx_c, n_eval // n_cls, replace=False).tolist())
    eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
    keep = np.setdiff1d(np.arange(len(targets)), eval_idx)

    import copy
    eval_base = copy.copy(train)
    eval_base.transform = dset_obj.test_dset.transform
    eval_base.data = train.data[eval_idx]
    eval_base.targets = targets[eval_idx].tolist()
    dset_obj.eval_dset = eval_base
    dset_obj.eval_orig_idx = eval_idx

    train.data = train.data[keep]
    train.targets = targets[keep].tolist()
    dset_obj.train_orig_idx = keep
    return dset_obj


def disjoint_indices(dset_obj, client_idx, n_per_client):
    """Indices (into the current train set) of client `client_idx` under the disjoint split.

    Uses the global numpy stream like the upstream splits do, so every rank draws the same split.
    The draw for all clients happens on every call, so client k's indices do not depend on which
    rank asks.
    """
    targets = np.array(dset_obj.train_dset.targets)
    all_idx = []
    for classes in DISJOINT_CLASSES:
        per = [n_per_client // len(classes)] * len(classes)
        for i in range(n_per_client - sum(per)):
            per[i] += 1
        idx = []
        for c, n in zip(classes, per):
            idx.extend(np.random.permutation(np.where(targets == c)[0])[:n].tolist())
        all_idx.append(idx)
    return all_idx if client_idx is None else all_idx[client_idx]


def plain_loader(dset, batch_size=512):
    # own generator: iterating it never touches the global torch random stream
    return DataLoader(dset, batch_size=batch_size, shuffle=False, generator=torch.Generator())


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def topk(model, loader, device):
    """(top1 %, top3 %) of `model` on `loader`. Leaves the model in eval mode, as upstream test() does."""
    model.eval()
    n = c1 = c3 = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        t3 = out.topk(3, dim=1).indices
        c1 += (t3[:, 0] == y).sum().item()
        c3 += (t3 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return 100.0 * c1 / n, 100.0 * c3 / n


class Scorer:
    """Eval and test loaders for one node."""

    def __init__(self, dset_obj):
        self.eval_loader = plain_loader(dset_obj.eval_dset)
        self.test_loader = plain_loader(dset_obj.test_dset)

    def __call__(self, model, device):
        e1, e3 = topk(model, self.eval_loader, device)
        t1, t3 = topk(model, self.test_loader, device)
        return {"eval_top1": e1, "eval_top3": e3, "test_top1": t1, "test_top3": t3}


# ---------------------------------------------------------------- stopping
class StopRule:
    """Train at least min_epochs; then stop once `patience` epochs pass without a new best eval
    top-1; never exceed max_epochs. Epochs are counted 1, 2, ... (epoch e = e rounds done)."""

    def __init__(self, a):
        self.min, self.pat, self.max = a["min_epochs"], a["patience"], a["max_epochs"]
        self.best = {}
        self.best_epoch = {}

    def update(self, name, epoch, value):
        """Record `value` for scored model `name` at `epoch`; return True if it is a new best."""
        if value > self.best.get(name, -1.0):
            self.best[name], self.best_epoch[name] = value, epoch
            return True
        return False

    def should_stop(self, epoch):
        if epoch >= self.max:
            return True
        if epoch < self.min:
            return False
        last = max(self.best_epoch.values()) if self.best_epoch else 0
        return epoch - last >= self.pat

    def stopped_by(self, epoch):
        return "cap" if epoch >= self.max else "rule"


# ---------------------------------------------------------------- records
def run_dir(config):
    return config["results_path"]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def record_run(config, dset_obj, extra=None):
    """Server/CENT side: run.json with config, commit and the eval indices."""
    rec = {"config": {k: v for k, v in config.items() if _jsonable(v)},
           "git_commit": git_commit(), "seed": config["seed"],
           "eval_orig_idx": dset_obj.eval_orig_idx.tolist()}
    rec.update(extra or {})
    write_json(os.path.join(run_dir(config), "run.json"), rec)


def record_client(config, dset_obj, client_idx, local_idx, class_counts):
    orig = dset_obj.train_orig_idx[np.asarray(local_idx, dtype=np.int64)].tolist()
    write_json(os.path.join(run_dir(config), "clients", f"client{client_idx}.json"),
               {"client": client_idx, "orig_idx": orig, "class_counts": class_counts})


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    torch.save(m.state_dict(), path)


def dream_path(config, epoch, who):
    return os.path.join(run_dir(config), "dreams", f"e{epoch:04d}_{who}.pt")


def save_dream(config, epoch, who, obj):
    p = dream_path(config, epoch, who)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in obj.items()}, p)


def soft_label_entropy(mean_logits):
    p = torch.softmax(mean_logits.float(), dim=1)
    return float(-(p * torch.log(p + 1e-12)).sum(dim=1).mean())
