"""Evaluate image-only recalibration on held-out doorway geometries."""
import argparse
import json
from pathlib import Path
import numpy as np
from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Camera, Unmeasurable
from aura.exploration.movement_calibration import CalibratingCamera
from aura.memory.evidence_ledger import EvidenceLedger
from aura.types import Source


def trial(seed, size, column, reverse, policy, ledger):
    world = ProceduralWorld('key',size=size,door_column=column,reverse_after_key=reverse,
                            max_steps=65,diagnostics=True)
    first,_ = world.reset(seed=seed)
    frames = [first]
    actions = []
    def step(action):
        image,_,done,truncated,_ = world.step(action)
        actions.append(action)
        frames.append(image)
        if done or truncated:
            raise Unmeasurable('episode ended during probe')
        return image
    camera = (CalibratingCamera if policy=='calibrated' else Camera)(first,step)
    reason = None
    try:
        passed = camera.measure(2)
    except Unmeasurable as exc:
        passed = False
        reason = str(exc)
    actual = world.agent == world.door
    if passed != actual:
        raise RuntimeError('Camera result disagrees with evaluator state')
    rid = f'{seed}:{size}:{column}:{reverse}:{policy}'
    ledger.append(rid,'movement_trial',Source.OBSERVED,dict(actions=actions,frames=np.stack(frames).tolist()))
    result = dict(seed=seed,size=size,column=column,reversal=reverse,policy=policy,
                  crossed=actual,steps=len(actions),reason=reason,trace=rid,
                  inferred_reversal=getattr(camera,'reversed',None))
    ledger.append(rid+':result','movement_result',Source.INFERRED,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True)
    ledger=EvidenceLedger(args.output/'evidence.sqlite')
    try:
        rows=[trial(seed,size,column,reverse,policy,ledger)
              for seed in range(53100,53110)
              for size,column in ((9,3),(9,5),(11,5),(11,6))
              for reverse in (False,True)
              for policy in ('calibrated','fixed')]
        assert ledger.verify()
    finally:
        ledger.close()
    totals={}
    for policy in ('calibrated','fixed'):
        for reverse in (False,True):
            part=[r for r in rows if r['policy']==policy and r['reversal']==reverse]
            totals[f'{policy}:{reverse}']=sum(r['crossed'] for r in part)
    assert totals == {'calibrated:False':40,'calibrated:True':40,'fixed:False':40,'fixed:True':0},totals
    assert all(r['inferred_reversal']==r['reversal'] for r in rows if r['policy']=='calibrated')
    (args.output/'results.json').write_text(json.dumps(dict(totals=totals,rows=rows),indent=2)+'\n')
    print(totals)


if __name__=='__main__':
    main()
