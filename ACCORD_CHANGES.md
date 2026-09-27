# Changes against upstream `aidecentralized/codream` @ 393fe6b

Branch `accord-repro`. One line per change: file, function, why.

## Fixes needed to run upstream at all

- `scheduler.py` imports and `algo_map`: removed `algos.dare`, `algos.distill_reps`,
  `algos.feddream_fast_independent`, which upstream deleted in 3118b19 ("only keep reqd algos") but
  still imports.
- `scheduler.py`: import of `algos.feddream` (plain CoDream) guarded, because it imports
  `algos/algos.py`, also deleted upstream. Plain CoDream is not runnable at 393fe6b; CoDream-fast is.

## Upstream vs paper (documented, not changed)

- Soft label: `FedDreamFastServer.add_dreams` averages the clients' logits; the paper's Eq. 5
  averages class probabilities.
- CoDream-fast dreams are `G(z)`: the server averages the clients' generators and latents.
