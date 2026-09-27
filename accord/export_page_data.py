"""M5: turn the grid's run directories into the data the results page expects.

usage: python accord/export_page_data.py GRID_DIR[,GRID_DIR2...] OUT_DIR [--cifar ROOT] [--dream-seed 4]

Several grid dirs may be given, comma-separated; a (split, arm, seed) run in a later dir replaces
the same run from an earlier one (used for the NaN-guarded CoDream-fast reruns).

Writes OUT_DIR/codream-data.json (the schema in the page's Implementation block) and
OUT_DIR/codream-exp/{examples,dreams}/<split>/*.png (32x32, upscaled only by the page's CSS).
Only what the page's placeholders show is exported.
"""
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image

SPLITS = ["iid", "dir01", "disjoint"]
ARMS = {"cent": ["CENT"], "fedavg": ["FedAvg"], "fast": ["CoDream-server", "CoDream-clients"]}


def runs(grids):
    out = {}
    for rj in [p for g in grids.split(",") for p in sorted(glob.glob(f"{g}/*/run.json"))]:
        d = os.path.dirname(rj)
        if not os.path.exists(f"{d}/done.json"):
            continue
        cfg = json.load(open(rj))["config"]
        split, arm = cfg["exp_id"].split("-")
        out.setdefault((split, arm), {})[cfg["seed"]] = d
    return out


def metrics(d):
    return [json.loads(l) for l in open(f"{d}/metrics.jsonl")]


def scored(d, arm):
    """{page arm name: (metrics key, stop-rule key)} for one run."""
    if arm == "fast":
        return {"CoDream-server": ("server", "server"), "CoDream-clients": ("clients_mean", "clients")}
    return {ARMS[arm][0]: ("server", "server")}


def save_png(t, path):
    """t: [3,32,32] float in [0,1] (generator output) or uint8 HWC array."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if torch.is_tensor(t):
        t = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(t).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid"); ap.add_argument("out")
    ap.add_argument("--cifar", default=os.environ.get("ACCORD_DATA", "./imgs/cifar10"))
    ap.add_argument("--dream-seed", type=int, default=4)
    a = ap.parse_args()
    R = runs(a.grid)
    D = {"scores": [], "curves": [], "counts": {}, "dreams": {}, "entropy": {}}

    for (split, arm), by_seed in sorted(R.items()):
        per = {}
        for seed, d in sorted(by_seed.items()):
            M = metrics(d)
            done = json.load(open(f"{d}/done.json"))
            for name, (mkey, rkey) in scored(d, arm).items():
                be = done["best_epoch"][rkey]
                row = [m for m in M if m["epoch"] == be][0][mkey]
                p = per.setdefault(name, {"top1": [], "top3": [], "curves": []})
                p["top1"].append(row["test_top1"]); p["top3"].append(row["test_top3"])
                p["curves"].append([m[mkey]["eval_top1"] for m in M])
        for name, p in per.items():
            D["scores"].append({"split": split, "arm": name, "seeds": sorted(by_seed),
                                "top1": p["top1"], "top3": p["top3"]})
            n = min(len(c) for c in p["curves"])
            D["curves"].append({"split": split, "arm": name, "epoch": list(range(1, n + 1)),
                                "top1": np.mean([c[:n] for c in p["curves"]], axis=0).round(3).tolist()})

    from torchvision.datasets import CIFAR10
    cifar = CIFAR10(root=a.cifar, train=True, download=False)
    targets = np.array(cifar.targets)
    for split in SPLITS:
        d = R.get((split, "fast"), {}).get(a.dream_seed)
        if d is None:
            continue
        clients = [json.load(open(f"{d}/clients/client{k}.json")) for k in range(4)]
        D["counts"][split] = [c["class_counts"] for c in clients]
        for k, c in enumerate(clients):
            idx = np.array(c["orig_idx"])
            for cls in range(10):
                hit = idx[targets[idx] == cls]
                if hit.size:
                    save_png(cifar.data[hit[0]], f"{a.out}/codream-exp/examples/{split}/client{k}_class{cls}.png")
        # dream log: every logged epoch that is a multiple of 10, plus the best epochs
        done = json.load(open(f"{d}/done.json"))
        keep = {e for e in done["best_epoch"].values()}
        epochs = sorted(int(os.path.basename(p)[1:5]) for p in glob.glob(f"{d}/dreams/e*_agg.pt"))
        epochs = [e for e in epochs if e % 10 == 0 or e in keep]
        probs = {}
        for e in epochs:
            A = torch.load(f"{d}/dreams/e{e:04d}_agg.pt")
            probs[str(e)] = {"agg": A["soft_label"].float().tolist()}
            for i in range(A["image"].shape[0]):
                save_png(A["image"][i], f"{a.out}/codream-exp/dreams/{split}/r{e}_agg_slot{i}.png")
            for k in range(4):
                C = torch.load(f"{d}/dreams/e{e:04d}_client{k}.pt")
                probs[str(e)][f"client{k}"] = C["proposal_probs"].float().tolist()
                for i in range(C["proposal"].shape[0]):
                    save_png(C["proposal"][i], f"{a.out}/codream-exp/dreams/{split}/r{e}_client{k}_slot{i}.png")
        D["dreams"][split] = {"rounds": epochs, "slots": 8, "probs": probs}
        M = metrics(d)
        D["entropy"][split] = {"round": [m["epoch"] for m in M],
                               "H": [round(m["soft_label_entropy"], 4) for m in M]}

    os.makedirs(a.out, exist_ok=True)
    json.dump(D, open(f"{a.out}/codream-data.json", "w"))
    print("wrote", f"{a.out}/codream-data.json", {k: len(v) for k, v in D.items()})


if __name__ == "__main__":
    main()
