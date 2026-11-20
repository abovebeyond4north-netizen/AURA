#!/usr/bin/env python3
"""Synthetic calibrator and conditional-variability diagnostics, not OOD certification."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from aura.world_model.uncertainty import UncertaintyEstimator


def experiment(seeds=range(5)):
    entries=[]
    for seed in seeds:
        rng=np.random.default_rng(seed+100)
        train=[rng.integers(0,12,(3,7,7),dtype=np.uint8) for _ in range(100)]
        validation=[rng.integers(0,12,(3,7,7),dtype=np.uint8) for _ in range(100)]
        shifted=[]
        for _ in range(100):
            x=rng.integers(0,12,(3,7,7),dtype=np.uint8)
            x[0]=rng.integers(220,256,(7,7),dtype=np.uint8)
            shifted.append(x)
        monitor=UncertaintyEstimator(None)
        monitor.fit_shift_detector(train,alpha=.05)
        fpr=float(np.mean([monitor.detect_distribution_shift(x) for x in validation]))
        tpr=float(np.mean([monitor.detect_distribution_shift(x) for x in shifted]))
        before=np.zeros((3,7,7),dtype=np.uint8)
        for i in range(40):
            monitor.record_observed_outcome(before,0,before)
            monitor.record_observed_outcome(before,1,np.full_like(before,255 if i%2 else 0))
        entries.append({'seed':seed,'in_distribution_false_alarm':fpr,
                        'shifted_detection_rate':tpr,
                        'deterministic_conditional_variance':monitor.conditional_outcome_variance(before,0),
                        'stochastic_conditional_variance':monitor.conditional_outcome_variance(before,1)})
    return {'experiment':'synthetic_no_label_shift_and_observed_variability',
            'seeds':list(seeds),'runs':entries,
            'mean_false_alarm':float(np.mean([e['in_distribution_false_alarm'] for e in entries])),
            'mean_shift_detection':float(np.mean([e['shifted_detection_rate'] for e in entries])),
            'limitations':['Synthetic color shift only; fixed-feature detector is not universal OOD detection',
                           'Conditional observed variability can be caused by unobserved state',
                           'Nominal alpha relies on exchangeability of the reference/calibration distribution']}

if __name__=='__main__':
    outcome=experiment()
    target=Path('sample_outputs/uncertainty_diagnostics.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(outcome,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in outcome.items() if k.startswith('mean_')},indent=2))
