# Changes against upstream `aidecentralized/codream` @ 393fe6b

Branch `accord-repro`. One line per change: file, function, why.

## Fixes needed to run upstream at all

- `scheduler.py` imports and `algo_map`: removed `algos.dare`, `algos.distill_reps`,
  `algos.feddream_fast_independent`, which upstream deleted in 3118b19 ("only keep reqd algos") but
  still imports.
- `scheduler.py`: import of `algos.feddream` (plain CoDream) guarded, because it imports
  `algos/algos.py`, also deleted upstream. Plain CoDream is not runnable at 393fe6b; CoDream-fast is.

## Additions (all inactive unless the config has an `accord` block)

- `utils/accord.py` (new): eval-set carve, disjoint split, top-1/top-3 scorer, stopping rule,
  run records, checkpoints, dream log. Uses no global random stream.
- `algos/base_class.py` `BaseNode.__init__`: carve the eval set, build the scorer.
- `algos/base_class.py` `BaseClient.set_data_parameters`: `non_iid_disjoint` branch; keep the
  client's indices (`_accord_local_idx`); new `accord_record`.
- `algos/centralized.py` `set_training_data`: `non_iid_disjoint` branch (union of the four
  clients); keep indices. New `accord_run_protocol`, entered from `run_protocol`.
- `algos/fl.py` `FedAvgClient.run_protocol`: stop on "STOP"; write indices on the first round.
  `FedAvgServer.accord_run_protocol` (new), entered from `run_protocol`.
- `algos/feddream_fast.py`: new tag `ACCORD = 21`. Client `fast_synthesize` keeps the generator
  output before augmentation (`gen_out`; same computation) and captures slots at the last local
  step; client `add_dreams` captures its logits on the aggregate; client `update_local_model`
  appends eval/test scores to the stats; client `run_protocol` stops on "STOP", receives the
  save flags after each round. Server `single_round_fast`/`add_dreams` capture the aggregate
  image, mean logits and the soft-label entropy of every batch;
  `FedDreamFastServer.accord_run_protocol` (new), entered from `run_protocol`.
- `configs/accord/grid.py` (new): the 9 cells over the upstream CIFAR10 IID configs.
- `tests/check_runs.py`, `tests/check_artifacts.py` (new): validators.

## Upstream behaviours kept (not in the paper)

- CENT (`CentralizedServer.train`) passes `epoch`, so `ModelUtils.adjust_learning_rate` sets lr
  0.1 / 0.01 after epoch 80 / 0.001 after 120, overriding `model_lr`.
- Upstream `test()` of CENT and FedAvg saves the model with the best **test** accuracy to
  `saved_models/`; ours select on the eval set (`ckpt/`).
- Plain CoDream (`algos/feddream.py`) is not runnable at 393fe6b.

## Upstream vs paper (documented, not changed)

- Soft label: `FedDreamFastServer.add_dreams` averages the clients' logits; the paper's Eq. 5
  averages class probabilities.
- CoDream-fast dreams are `G(z)`: the server averages the clients' generators and latents.
