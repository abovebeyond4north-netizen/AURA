"""Controlled cross-environment study with independent final evaluation sets."""
import argparse
import json
from pathlib import Path

import numpy as np
from aura.cognition.concept_library import ConceptLibrary
from aura.memory.evidence_ledger import EvidenceLedger
from aura.types import Source

# Signals correspond to empirical features. Environment identity and the
# mechanism generating the outcome are withheld from the learner.
SIGNALS = np.array([-3.,-1.,1.,3.])
PHASES = [
    ('initial', [0,0,1,1], 'create'),
    ('changed', [0,.6,1,1], 'split'),
    ('simplified', [0,0,1,1], 'merge'),
    ('uninformative', [.5,.5,.5,.5], 'retire'),
]


def run(seed, ledger):
    learner = ConceptLibrary(ledger, str(seed))
    rng = np.random.default_rng(seed)
    results = []

    def collect(stage, role, table, offset, jitter):
        features = []
        outcomes = []
        ids = []
        order = rng.permutation(np.repeat(np.arange(4),32))
        for index, state in enumerate(order):
            feature = [float(SIGNALS[state]+offset+rng.uniform(-jitter,jitter))]
            result = table[state]
            rid = f'concept-sample:{seed}:{stage}:{role}:{index}'
            ledger.append(rid, 'concept_sample', Source.OBSERVED,
                          {'features': feature, 'outcome': result, 'environment': role})
            features.append(feature)
            outcomes.append(result)
            ids.append(rid)
        return ids,np.array(features),np.array(outcomes)

    for stage, table, expected in PHASES:
        train, x, y = collect(stage,'source',table,0,.08)
        validation, *_ = collect(stage,'source-validation',table,0,.12)
        transfer, *_ = collect(stage,'transfer-development',table,.17,.18)
        incumbent = learner.model
        result = learner.revise(stage,train,validation,transfer)
        restored = ConceptLibrary(ledger, str(seed))
        if learner.model is None:
            if restored.model is not None:
                raise AssertionError('Retired prototypes unexpectedly restored')
        else:
            np.testing.assert_array_equal(restored.model.predict(x), learner.model.predict(x))
        learner = restored
        if result['accepted'] != expected:
            raise AssertionError((stage, expected, result['accepted']))
        # Generate the final environment observations only after selection.
        _, features, targets = collect(stage,'transfer-final',table,-.28,.24)
        old = incumbent.predict(features) if incumbent else y.mean()
        new = learner.model.predict(features) if learner.model else y.mean()
        mse = lambda prediction: float(np.mean((prediction-targets)**2))
        before, after, constant = mse(old), mse(new), mse(y.mean())
        if after > before + 1e-10 or (stage != 'uninformative' and after >= constant):
            raise AssertionError((stage,before,after,constant))
        results.append(dict(stage=stage,operation=expected,prior_mse=before,
                            final_mse=after,constant_mse=constant))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    ledger = EvidenceLedger(args.output/'evidence.sqlite')
    try:
        results = {seed:run(seed,ledger) for seed in (1907,3019,5167,8009,10487)}
        if not ledger.verify():
            raise RuntimeError('Evidence ledger did not verify')
    finally:
        ledger.close()
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Passed 20 lifecycle decisions and final transfer checks')


if __name__ == '__main__':
    main()
