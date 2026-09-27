"""Validate accord run directories (M1 checks 1, 4, 5 and M4 check 3).

usage: python tests/check_runs.py RUN_DIR [RUN_DIR ...] [--ckpt]
Run dirs of the same split and seed are compared with each other (same client indices across arms,
CENT data = union of the client data). Exits non-zero on the first failed check.
"""
import glob, json, os, sys
import numpy as np
import torch
from torchvision.datasets import CIFAR10

DISJOINT = [[0, 1], [2, 3], [4, 5, 6], [7, 8, 9]]


def fail(msg):
    print("FAIL", msg); sys.exit(1)


def load(rd):
    run = json.load(open(f"{rd}/run.json"))
    clients = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{rd}/clients/*.json")}
    return run, clients


def check_one(rd, targets):
    run, clients = load(rd)
    cfg = run["config"]
    ev = np.array(run["eval_orig_idx"])
    if len(ev) != 5000 or len(set(ev)) != 5000: fail(f"{rd}: eval set size {len(ev)}")
    if np.bincount(targets[ev], minlength=10).tolist() != [500] * 10: fail(f"{rd}: eval not 500/class")
    split = cfg["exp_id"].split("-")[0]
    if "cent" in clients:
        idx = np.array(clients["cent"]["orig_idx"])
        if len(idx) != 4000 or len(set(idx)) != 4000: fail(f"{rd}: CENT size {len(idx)}")
        if np.intersect1d(idx, ev).size: fail(f"{rd}: CENT data overlaps eval")
        return split, cfg["seed"], {"cent": idx}
    if sorted(clients) != [f"client{k}" for k in range(4)]: fail(f"{rd}: clients {sorted(clients)}")
    out = {}
    for k in range(4):
        idx = np.array(clients[f"client{k}"]["orig_idx"])
        if len(idx) != 1000 or len(set(idx)) != 1000: fail(f"{rd}: client{k} size {len(idx)}")
        if np.intersect1d(idx, ev).size: fail(f"{rd}: client{k} overlaps eval")
        cc = np.bincount(targets[idx], minlength=10)
        if cc.tolist() != clients[f"client{k}"]["class_counts"]: fail(f"{rd}: client{k} class counts mismatch")
        if split == "disjoint" and set(np.nonzero(cc)[0]) != set(DISJOINT[k]): fail(f"{rd}: client{k} classes {np.nonzero(cc)[0]}")
        if split == "iid" and cc.min() < 60: fail(f"{rd}: IID client{k} class counts {cc}")
        out[k] = idx
    for a in range(4):
        for b in range(a + 1, 4):
            if np.intersect1d(out[a], out[b]).size: fail(f"{rd}: client{a} and client{b} share images")
    return split, cfg["seed"], {"clients": np.sort(np.concatenate([out[k] for k in range(4)])),
                                 "per_client": [np.sort(out[k]) for k in range(4)]}


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = os.environ.get("ACCORD_DATA", "./imgs/cifar10")
    targets = np.array(CIFAR10(root=root, train=True, download=False).targets)
    groups = {}
    for rd in dirs:
        split, seed, rec = check_one(rd, targets)
        groups.setdefault((split, seed), []).append((rd, rec))
        print("ok  ", rd)
    for (split, seed), recs in groups.items():
        fed = [r for r in recs if "clients" in r[1]]
        cent = [r for r in recs if "cent" in r[1]]
        for rd, r in fed[1:]:
            for k in range(4):
                if not np.array_equal(r["per_client"][k], fed[0][1]["per_client"][k]):
                    fail(f"{split} seed {seed}: client{k} data differs between {fed[0][0]} and {rd}")
        for rd, r in cent:
            if fed and not np.array_equal(np.sort(r["cent"]), fed[0][1]["clients"]):
                fail(f"{split} seed {seed}: CENT data != union of client data ({rd})")
        print(f"ok   {split} seed {seed}: {len(recs)} runs share one split")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
