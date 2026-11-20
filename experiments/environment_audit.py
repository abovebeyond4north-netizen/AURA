"""Construct and independently replay reachability witnesses across layout splits."""
import argparse
import json
from pathlib import Path
from aura.environments.procedural_world import ProceduralWorld
from aura.environments.task_generator import layout_spec
from protected_verifier.solvability import evaluator_solution


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for split in ('train','dev','protected'):
        for index in range(30):
            spec=layout_spec(split,index)
            kwargs={k:v for k,v in spec.items() if k!='seed'}
            seed=spec['seed']
            env=ProceduralWorld(**kwargs,max_steps=65,diagnostics=True)
            env.reset(seed=seed)
            path=evaluator_solution(env)
            if path is None:
                raise RuntimeError(f'No reachability witness for {spec}')
            replay=ProceduralWorld(**kwargs,max_steps=65,noise=.2)
            image,meta=replay.reset(seed=seed)
            if not replay.observation_space.contains(image) or 'rule' in meta:
                raise RuntimeError('Invalid observation envelope')
            try:
                replay.diagnostic_state()
            except PermissionError:
                pass
            else:
                raise RuntimeError('Diagnostics exposed through learner instance')
            for i,action in enumerate(path):
                image,reward,done,truncated,info=replay.step(action)
                if truncated or done!=(i==len(path)-1) or 'rule' in info:
                    raise RuntimeError('Witness failed during replay')
            rows.append(dict(split=split,**spec,actions=path))
    with args.output.open('x') as file:
        json.dump(rows,file,indent=2)
    print(f'{len(rows)} replayed, longest witness {max(len(r["actions"]) for r in rows)} actions')


if __name__=='__main__':
    main()
