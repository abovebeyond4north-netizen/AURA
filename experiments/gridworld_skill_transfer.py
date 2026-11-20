"""Discover doorway action sequences, then test reuse on fresh grid layouts."""
import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Camera, MOVES, offset
from aura.exploration.visual_frontier import visual_displacement
from aura.memory.evidence_ledger import EvidenceLedger
from aura.skills.door_option import OptionReliability, learn, resolve
from aura.types import Source


def prepare(size, column, seed, keyed=True):
    world = ProceduralWorld('key', size=size, max_steps=80, diagnostics=True)
    image, _ = world.reset(seed=seed)
    if world.door[1] != column:
        world.walls.add(world.door)
        world.door = (size//2,column)
        world.walls.remove(world.door)
        image = world._observe()
    actions = []
    frames = [image]
    def step(action):
        actions.append(action)
        image = world.step(action)[0]
        frames.append(image)
        return image
    camera = Camera(image, step)
    direction = camera.prepare(2 if keyed else 0)
    return world, camera, direction, actions, frames


def probe(camera, direction, sequence):
    delta = MOVES[direction]
    before = camera.image
    if camera.at_pixel(delta) != 'door':
        return False
    middle = camera.step(resolve(sequence[0],direction))
    camera.image = middle
    after = camera.step(resolve(sequence[1],direction))
    return (visual_displacement(before,middle)==(0,0)
            and camera.at_pixel(delta)=='empty'
            and visual_displacement(middle,after)==delta)


def finish(camera, world):
    # Compose passage with navigation to a goal observed from the doorway.
    for row in range(-3,4):
        for col in range(-3,4):
            if camera.at_pixel((row,col)) == 'goal':
                goal = offset(camera.position, (row,col))
                camera.map[goal] = 'empty'
                route = camera.paths().get(goal)
                if route is not None:
                    camera.travel(route)
                    return world.diagnostic_state()['agent'] == world.goal
    return False


def run(ledger):
    summaries = []
    for seed in (157,411,829,1013,1709):
        rng = np.random.default_rng(seed)
        samples = []
        # Equal exposure to all primitive pairs at a common visual precondition.
        sequences = list(product((0,'forward',2,3,4),repeat=2))*3
        rng.shuffle(sequences)
        def save_trial(id, world, actions, frames, start, effect):
            observed = world.diagnostic_state()['agent'] == world.door
            if effect != observed:
                raise AssertionError('Camera effect and physical door passage differ')
            payload = {'actions':actions[start:], 'frames':np.stack(frames[start:]).tolist(),
                       'steps_before_option':start}
            ledger.append(id,'skill_trial',Source.OBSERVED,payload)
            ledger.append(id+':effect','skill_outcome',Source.INFERRED,
                          {'passed':effect,'evidence_id':id})
        for i, seq in enumerate(sequences):
            world,camera,direction,actions,frames = prepare(9,4,seed*1000+i)
            start = len(actions)
            assert direction == 1
            result = probe(camera,direction,seq)
            rid = f'{seed}:train:{i}'
            # A paired action may enter the door without matching the full
            # option signature. Keep the raw trajectory in all cases.
            ledger.append(rid,'skill_trial',Source.OBSERVED,
                          {'actions':actions[start:], 'frames':np.stack(frames[start:]).tolist()})
            if result and world.diagnostic_state()['agent'] != world.door:
                raise AssertionError('False positive visual effect for door option')
            ledger.append(rid+':effect','skill_outcome',Source.INFERRED,
                          {'passed':result,'evidence_id':rid})
            samples.append((seq,result,rid))
        option = learn(samples)
        if option is None:
            raise AssertionError('No evidence-supported option found')
        ledger.append(f'{seed}:option','learned_option',Source.INFERRED,asdict(option))
        wins = dict(reuse=0,random=0,blocked=0)
        goals = dict(reuse=0,random=0)
        reliability = OptionReliability()
        for i in range(20):
            size,column = [(11,4),(11,6),(9,3),(9,5)][i%4]
            for arm in wins:
                world,camera,direction,actions,frames = prepare(size,column,9000000+seed*100+i,arm!='blocked')
                start=len(actions)
                if arm=='random':
                    primitives = list(range(5))
                    seq = tuple(primitives[int(rng.integers(5))] for _ in range(2))
                    # Count physical passage for the baseline, even if its
                    # actions differ from the learned option's signature.
                    for action in seq:
                        camera.step(action)
                    effect = world.diagnostic_state()['agent'] == world.door
                    if effect:
                        camera.position = offset(camera.position,MOVES[direction])
                        camera._observe(frames[-1])
                else:
                    effect = option.execute(camera,direction)
                save_trial(f'{seed}:{arm}:{i}',world,actions,frames,start,effect)
                wins[arm]+=effect
                if arm == 'blocked':
                    reliability.record('key-missing',f'{seed}:{arm}:{i}',effect)
                if arm in goals and effect:
                    goals[arm] += finish(camera, world)
                    ledger.append(f'{seed}:{arm}:{i}:plan','skill_plan_observation',Source.OBSERVED,
                                  {'actions':actions[start:], 'frames':np.stack(frames[start:]).tolist(),
                                   'terminated_at_goal':world.diagnostic_state()['agent']==world.goal})
        assert not reliability.available('key-missing') and reliability.available('key-present')
        ledger.append(f'{seed}:withdraw','skill_withdrawal',Source.INFERRED,
                      {'context':'key-missing','evidence_ids':list(reliability.outcomes['key-missing'])})
        assert goals['reuse'] > goals['random'],goals
        assert wins['reuse']==20 and wins['blocked']==0 and wins['random']<20,wins
        summaries.append(dict(seed=seed,training_episodes=len(samples),evaluation_per_arm=20,wins=wins,composed_goals=goals))
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True)
    ledger=EvidenceLedger(args.output/'evidence.sqlite')
    try:
        rows=run(ledger)
        assert ledger.verify()
    finally:
        ledger.close()
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(rows)


if __name__ == '__main__':
    main()
