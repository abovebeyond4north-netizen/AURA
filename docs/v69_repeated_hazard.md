# v69 repeated unsignaled resets

Public controlled stress experiment; no controller changes or promotion. Two seeds (98173, 98174), paired fixed/adaptive hazard, sizes 9/11, two mapping indices, noise 0/8%, reset chances 0/4/12% per action after six key-stage steps. A reset samples uniformly and may reproduce the current mapping. The reset stream is seeded externally to the learner; realized exposure depends on episode length. The 4% condition had only two reset events per controller across both seeds, so is weak evidence. Both controllers reached 40/48 goals: 16/16 at 0%, 16/16 at 4%, 8/16 at 12%. Both recorded four incorrect mapping identifications and zero audited pose errors. The mapping may change during recovery probes, violating the controller assumption. No adaptive advantage. Only 48 fresh records replayed, not independent evaluation. Nine focused tests passed; previous register unchanged.

Reproduce from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.repeated_hazard_study --seed 98173 --output /tmp/v69-development.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.repeated_hazard_study --seed 98174 --output /tmp/v69-fresh.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_adaptive_hazard.py tests/test_hazard_dynamics.py
```

Next: propagate reset hypotheses during each recovery probe; preserve hard pose certification and test mapping errors explicitly. Gymnasium import remains unavailable.
