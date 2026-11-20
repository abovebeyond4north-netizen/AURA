# v62 — Joint movement mapping and spatial calibration

Experimental mixin retains both normal/reversed controls and blocked/moved outcomes when the original two-clear-sided axis probe is ambiguous. It ranks safe raw probes using mapped stationary contrasts, uses static compatibility to eliminate states, and commits only a unique joint estimate with at least 12 map matches (original successful fast path retained). Recovery is bounded at four raw probes and 12 retained states; unrecognized/dynamic targets are excluded from extra probes. The map is not updated during unresolved recovery. A temporary local empty tile represents the just-collected key. This is a hand-engineered finite belief mechanism, not learned probabilistic inference. It assumes the existing binary reversal model, known palette, static map and off-palette corruption.

Development (previously inspected seed 93200, 9:6,11:3,13:9,15:4,17:10, 0/3/8% corruption): 30 matched trials per arm, v60 18, calibrated v60 18, v61 14, calibrated v61 17 goals. Nine v61 calibration-specific failures become zero; downstream failures persist. Development rows predate the mapping audit addition. No pose errors recorded.

Fresh public seeds 94100 (9:5,11:6,13:4,15:10,17:8) and 94200 (9:3,11:7,13:10,15:5,17:12), 0/3/8% noise. Stress public seeds 94300 (9:4,11:4,13:6,15:7,17:5) and 94400 (9:5,11:8,13:9,15:9,17:9), 8/12/15% noise. Both control mappings, same 65 actions. Each pair of batches has 60 trials per arm. No independent secret holdout.

| Controller | Fresh goals | Stress goals | Fresh action cost* | Stress action cost* |
|---|---:|---:|---:|---:|
| v60 |40|37|2828|2747|
| Calibrated v60 |40|38|2828|2714|
| v61 |36|45|2452|2198|
| Calibrated v61 |36|45|2452|2198|

*Failed episodes charged full budget. One stress gain, zero losses; no fresh gain and no observed success losses. No pose errors across 600 original records; zero mapping errors among 439 completed fresh/stress calibrations (44+44+60+60+55+56+60+60 = 439). Mapping audit only checks completed calibration calls, not intermediate or aborted calls. 480 replay records compare equal as parsed JSON; 240 stress records byte-identical. Fresh serialization changed integer zero to float zero when adding the optional noise CLI argument. 18 focused regression tests passed. Full suite not run; no new commit or promotion, acceptance stays 13/31. Next: examine remaining movement contradictions and unresolved pose traces; avoid further calibration tuning without new evidence.

Reproduction: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.adaptive_calibration_study --seed 94100 --layouts 9:5,11:6,13:4,15:10,17:8 --output out.json` (repeat with other listed seeds/layouts; stress adds `--noise .08,.12,.15`).
