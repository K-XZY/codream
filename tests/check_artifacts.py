"""M1 checks 4 and 5 on one run dir: checkpoint round-trip and dream-log consistency.

usage: python tests/check_artifacts.py RUN_DIR   (run from the repo root, on a GPU box)
"""
import glob, json, sys
import torch
from utils.data_utils import get_dataset
from utils.model_utils import ModelUtils
from utils import accord


def main(rd):
    run = json.load(open(f"{rd}/run.json"))
    cfg = run["config"]
    lines = [json.loads(l) for l in open(f"{rd}/metrics.jsonl")]
    done = json.load(open(f"{rd}/done.json"))
    # check 4: best checkpoint re-scores to the logged numbers
    dset = get_dataset("cifar10", cfg["dpath"])
    accord.carve_eval(dset, cfg)
    if dset.eval_orig_idx.tolist() != run["eval_orig_idx"]:
        print("FAIL eval indices not reproducible"); sys.exit(1)
    dev = torch.device("cuda:0")
    model = ModelUtils.get_model("resnet18", "cifar10", dev, [0], num_classes=10)
    sd = torch.load(f"{rd}/ckpt/server_best.pt", map_location=dev)
    (model.module if hasattr(model, "module") else model).load_state_dict(sd)
    sc = accord.Scorer(dset)(model, dev)
    be = done["best_epoch"]["server"]
    logged = [l for l in lines if l["epoch"] == be][0]["server"]
    for k in sc:
        if abs(sc[k] - logged[k]) > 1e-6:
            print(f"FAIL ckpt round-trip {k}: {sc[k]} vs logged {logged[k]}"); sys.exit(1)
    print(f"ok   checkpoint round-trip at epoch {be}: {sc}")
    # check 5: aggregate soft label = softmax of the mean of the clients' logits
    for agg in sorted(glob.glob(f"{rd}/dreams/e*_agg.pt")):
        e = agg.split("/")[-1][:5]
        A = torch.load(agg)
        C = [torch.load(f"{rd}/dreams/{e}_client{k}.pt") for k in range(4)]
        mean = torch.stack([c["logits_on_aggregate"] for c in C]).mean(0)
        if not torch.allclose(mean, A["mean_logits"], atol=1e-5):
            print(f"FAIL {e}: mean of client logits != saved aggregate logits"); sys.exit(1)
        if not torch.allclose(torch.softmax(mean.float(), 1), A["soft_label"], atol=1e-6):
            print(f"FAIL {e}: soft label != softmax(mean logits)"); sys.exit(1)
        for c in C:
            if c["proposal"].shape != A["image"].shape: print(f"FAIL {e}: shapes"); sys.exit(1)
        print(f"ok   dream log {e}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1])
