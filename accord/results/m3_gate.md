# M3 gate: seed 4 against the paper's Table 1 (CIFAR10, 4 clients)

Job 6905133 (Isambard, `workq`), code `7450dd1`, completed 2026-09-28 00:07 UTC. Each entry is the
test accuracy of the checkpoint with the best eval top-1 (5000 held-out training images); the
paper does not say how it selects its checkpoint, nor which model its CoDream column scores.
Pass rule (plan): IID within 5 pp; Dir(0.1) within max(paper std, 5 pp).

| split | arm | test top-1 | test top-3 | best epoch of run (stop) | paper | gate |
|---|---|---|---|---|---|---|
| IID | CENT | 67.94 | 91.03 | 143 of 400 (rule) | 65.7 ± 2.9 | pass |
| IID | FedAvg | 64.68 | 87.78 | 381 of 401 (rule) | 67.2 ± 0.4 | pass |
| IID | CoDream-server | 66.73 | 90.48 | 331 of 400 (rule) | 69.5 ± 0.3 | pass |
| IID | CoDream-clients | 68.86 | 90.92 | 271 of 400 (rule) | 69.5 ± 0.3 | pass |
| Dir(0.1) | CENT | 45.95 | 74.30 | 114 of 400 (rule) | 45.5 ± 6.8 | pass |
| Dir(0.1) | FedAvg | 42.04 | 72.00 | 236 of 400 (rule) | 34.8 ± 8.3 | pass |
| Dir(0.1) | CoDream-server | 33.28 | 65.34 | 116 of 400 (rule) | 36.6 ± 8.4 | pass* |
| Dir(0.1) | CoDream-clients | 34.33 | 67.51 | 116 of 400 (rule) | 36.6 ± 8.4 | pass* |

**Verdict: reproduced.** Every IID entry is within 5 pp of the paper; CoDream-clients is 0.6 pp
below it, CoDream-server 2.8 pp, FedAvg 2.5 pp, CENT 2.2 pp above.

\* **Dir(0.1) CoDream-fast is not a clean result.** From epoch 120 the dreams of this run are
non-finite and every model falls to chance; the entry is the checkpoint of epoch 116, just before.
Every non-IID CoDream-fast run of the grid does the same (first non-finite dream between epochs
19 and 120), no IID run does. The page's non-IID CoDream-fast rows therefore come from reruns
with the NaN guard (ACCORD_CHANGES.md; Kevin, 2026-09-28). Those reruns show that restoring the
generator does not stop the failure: from round 15, when the adversarial term of the dream
objective switches on (`FedDreamFastClient.fast_synthesize`, `self.round >= 15`), every batch goes
non-finite again. Being diagnosed.
