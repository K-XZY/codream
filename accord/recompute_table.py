"""M5 check 1: recompute every accuracy-table entry straight from metrics.jsonl and compare with
the exported JSON. Independent of export_page_data.py: it re-derives the best epoch from the
eval column instead of trusting done.json.

usage: python accord/recompute_table.py GRID_DIR OUT_DIR/codream-data.json
"""
import glob, json, sys


def main(grid, data):
    D = json.load(open(data))
    got = {(s["split"], s["arm"]): s for s in D["scores"]}
    n = 0
    for rj in glob.glob(f"{grid}/*/run.json"):
        d = rj[:-len("/run.json")]
        cfg = json.load(open(rj))["config"]
        split, arm = cfg["exp_id"].split("-")
        M = [json.loads(l) for l in open(f"{d}/metrics.jsonl")]
        keys = {"fast": [("CoDream-server", "server"), ("CoDream-clients", "clients_mean")],
                "fedavg": [("FedAvg", "server")], "cent": [("CENT", "server")]}[arm]
        for name, k in keys:
            best = max(M, key=lambda m: (m[k]["eval_top1"], -m["epoch"]))  # first epoch reaching the max
            s = got[(split, name)]
            i = s["seeds"].index(cfg["seed"])
            if abs(s["top1"][i] - best[k]["test_top1"]) > 1e-9 or abs(s["top3"][i] - best[k]["test_top3"]) > 1e-9:
                print(f"MISMATCH {split} {name} seed {cfg['seed']}: json {s['top1'][i]}/{s['top3'][i]} "
                      f"recomputed {best[k]['test_top1']}/{best[k]['test_top3']} (epoch {best['epoch']})")
                sys.exit(1)
            n += 1
    print(f"ALL {n} TABLE ENTRIES MATCH")


if __name__ == "__main__":
    main(*sys.argv[1:3])
