"""Illustrative epoch-0 frame for Figure 4 (not logged in any run).

A freshly initialised generator of the authors' architecture (utils/generator.py, nz=256, ngf=64)
applied to a batch of 256 standard-normal latents, in train mode as at the first dream batch.
At epoch 0 every client starts from the same generator and latents, so all rows are identical,
and no client has scored the dreams, so there are no class-distribution bars.

usage: python accord/step0_tiles.py PAGE_DIR   (PAGE_DIR = accord/results/page)
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from utils.generator import Generator

page = sys.argv[1]
torch.manual_seed(0)
g = Generator(nz=256, ngf=64, img_size=32, nc=3)
g.train()
with torch.no_grad():
    x = g(torch.randn(256, 256))[:8]
D = json.load(open(f"{page}/codream-data.json"))
for split, d in D["dreams"].items():
    for i in range(8):
        img = (x[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        for row in [f"client{k}" for k in range(4)] + ["agg"]:
            p = f"{page}/codream-exp/dreams/{split}/r0_{row}_slot{i}.png"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(img).save(p)
    if 0 not in d["rounds"]:
        d["rounds"] = [0] + d["rounds"]
    d["illustrative_rounds"] = [0]
json.dump(D, open(f"{page}/codream-data.json", "w"))
print("added epoch-0 illustration to", list(D["dreams"]))
