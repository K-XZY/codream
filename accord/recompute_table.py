"""M5 check 1: recompute every accuracy-table entry straight from metrics.jsonl and compare with
the exported JSON. Independent of export_page_data.py: it re-derives the best epoch from the
eval column instead of trusting done.json.

usage: python accord/recompute_table.py GRID_DIR[,GRID_DIR2...] OUT_DIR/codream-data.json [split:arm,...to skip]
(later dirs override the same split/arm/seed, as in export_page_data.py)
"""
import glob, json, sys


def main(grid, data, skip=""):
    skip = {tuple(c.split(":")) for c in skip.split(",") if c}
    D = json.load(open(data))
    got = {(s["split"], s["arm"]): s for s in D["scores"]}
    n = 0
    chosen = {}
    for rj in [p for g in grid.split(",") for p in sorted(glob.glob(f"{g}/*/run.json"))]:
        d = rj[:-len("/run.json")]
        if not glob.glob(f"{d}/done.json"):
            continue
        cfg = json.load(open(rj))["config"]
        split, arm = cfg["exp_id"].split("-")
        if (split, arm) not in skip:
            chosen[(split, arm, cfg["seed"])] = (d, cfg)
    for (split, arm, _), (d, cfg) in sorted(chosen.items()):
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
    main(*sys.argv[1:4])
