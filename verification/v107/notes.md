# v107 regression completion and evidence reconciliation

Eight disjoint six-test partitions passed with exit 0; all 48 remaining IDs are verified. Reconciliation checks the original 310-ID collection, successful batch logs, duplicate credit, five opt-in namespace results, and unchanged runtime/test/config/script paths from v101. 310/310 partitioned coverage; not a single-process run. All 1048 inherited manifest hashes match. Archive additionally retains both prior manifest files that were not themselves covered by the v106 manifest. No functional code or acceptance changes.

Historical bundle ancestry reconfirmed; 4acadba remains absent. Gymnasium and TensorBoard unavailable. Section 28 still requires OS matrix and quality gates. Section 25 independent evaluation outstanding. No new behavior experiments or promotion.

Reproduce in a disposable copy: `python verification/v107/reproduce.py` (or bounded continuation `python verification/v107/reproduce.py 18`, `python verification/v107/reproduce.py 36 18`, `python verification/v107/reproduce.py 48 36`). Reconcile using `python verification/v107/reconcile.py`.
