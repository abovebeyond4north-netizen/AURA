# v67 — Integrated regression recovery

No controller or verifier implementation was changed. The v66 manifest was checked before testing: 572/572 entries matched. Local Git history establishes that f582278, 2c8f161 and f0ab324 are ancestors of integration commit 4acadba0804d6d7cc7815fe3a1195ac5bb80424a. Comparing all 406 tracked files from that commit with the v66 source gives 404 exact matches, two updated documentation files and no missing files. Both continuation evidence directories are preserved.

The monolithic suite repeatedly stopped without a terminal summary within execution limits; partial logs are retained. A disjoint partition of the same collected 258 tests completed:

- Non-verifier: 208 passed, 5 skipped, 5 subtests passed (26.44 s).
- Verifier first partition: 35 passed, 10 deselected (32.19 s).
- Verifier complementary partition: 10 passed, 35 deselected (4.62 s).

Total: 253 passed, 5 skipped, plus 5 passing subtests; all three partition commands exited zero. This is a partitioned local regression, not a successful monolithic invocation. No claim that the interrupted runs indicate a test defect. Python 3.13.5 CPU, torch 2.10.0+cpu, numpy 2.3.5, scipy 1.17.0, pydantic 2.13.5, PyYAML 6.0.3, matplotlib 3.10.8, pytest 9.0.2; Gymnasium and TensorBoard absent. Independent evaluation remains unavailable in this run. Section 03, 08 and 28 remain partial; status remains 13/31 locally verified, not full specification completion.

## Reproduction

From source root, export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Run:

```sh
python -m pytest -q -o addopts='' --ignore=tests/test_verifier.py
python -m pytest -q -o addopts='' tests/test_verifier.py -k 'not (seccomp_denies_legacy_creat_syscall or seccomp_denies_write_capable or seccomp_denies_memfd or seccomp_denies_openat2 or seccomp_denies_xattr or runner_)'
python -m pytest -q -o addopts='' tests/test_verifier.py -k 'seccomp_denies_legacy_creat_syscall or seccomp_denies_write_capable or seccomp_denies_memfd or seccomp_denies_openat2 or seccomp_denies_xattr or runner_'
```

Next: verify installed Gymnasium conformance in a suitable runtime, and implement/evaluate adaptive hazard estimation on disjoint task schedules. No promotion.
