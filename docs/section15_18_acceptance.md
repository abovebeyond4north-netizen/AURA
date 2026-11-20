# Sections 15 and 18 · Meta-learning and computation allocation

Section 15 evaluated the existing meta-controller as a scheduler over three learning rates. Across 32 paired linear prediction problems, each policy ran exactly 24 gradient updates. Adaptive scheduling improved test MSE on all 32 seeds versus fixed round robin. Mean test MSE was 0.6820 adaptive, 0.8896 fixed, and 0.9773 uniform random. The paired bootstrap 95% interval for the fixed-minus-adaptive MSE difference was [0.1657, 0.2541].

Section 18 evaluated the existing value-of-computation allocator with small and large training batches. The budget was 64 units, each representing 16 sample presentations. Full batches cost eight units. The allocator used development loss reduction with a charge for computation and improved test MSE on all 32 seeds versus an always-large-batch policy. Mean test MSE was 0.2827 adaptive, 0.5969 fixed and 0.2962 random. The paired 95% bootstrap interval for the improvement versus the fixed policy was [0.2463, 0.3933]. The interval versus random included zero. Resource matching applies to training sample presentations, excluding development scoring and policy overhead.

Final test data were unavailable to the scheduling policies. Both experiments demonstrate local controlled acceptance for the existing schedulers. They do not establish end-to-end gridworld gains, global optimality or independently validated recursive improvement.

Run these commands from the source directory with unused output paths:

```sh
python -m experiments.meta_allocation --output meta_results.json
python -m experiments.computation_tradeoff --output cost_results.json
```

The complete run traces and held-out results are in `verification/section15_18/`. Seven focused door and evidence tests passed. The full pytest suite is unavailable in this runtime.
