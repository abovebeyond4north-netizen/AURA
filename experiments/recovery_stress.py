"""Public stress diagnostic: paired cameras on previously untested sizes/noise."""
import argparse
import json
from pathlib import Path
from experiments.goal_acquisition import trial
from aura.exploration.goal_search import PaletteGoalCamera, RecoveryGoalCamera


def run():
    return [dict(arm=arm, **trial(seed,size,col,rev,noise,'frontier',cls))
            for seed in range(96100,96108)
            for size,col in ((9,3),(11,5),(13,4),(13,8))
            for rev in (False,True) for noise in (.03,.08,.15)
            for arm,cls in [('baseline',PaletteGoalCamera),('recovery',RecoveryGoalCamera)]]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as f:
        json.dump(run(),f)
