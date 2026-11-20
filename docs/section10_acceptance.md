# Section 10 · Evidence and memory

The local acceptance test exercises observation persistence, contradictory semantic revisions, and procedural records with both successful and failed outcomes. The ledger rejects imagined records presented as experience or door observations. Earlier conclusions remain available when contrary evidence reduces confidence.

The checkpoint covers 858 SQLite records from the real Section 09 hidden-rule investigation, including observations, derived measurements, and inference revisions. The receipt commits to the entire ordered prefix, including timestamps. It stays valid when new records are appended and detects truncation, timestamp changes, or replacement with a different internally consistent ledger. The verifier checks the supplied receipt against a separately held SHA-256 pin before reading the database. Keep the pin outside the learner's storage. The published conversation pins this concrete checkpoint. The underlying SQLite file still depends on filesystem access controls, and the verifier covers only the anchored prefix.

From the source directory:

```sh
python -m unittest discover -s tests -p test_evidence_anchor.py -v
python -m aura.memory.anchor verify verification/section10/grid-ledger.sqlite verification/section10/receipt.json --pin RECEIPT_SHA256_FROM_CONVERSATION
```

Both focused memory tests and all five Section 09 regression tests passed. Full pytest execution remains unavailable in this runtime.
