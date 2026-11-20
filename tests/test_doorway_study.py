import numpy as np
import pytest
from experiments.doorway_study import DoorwayWorld, demonstration, TRAIN, RESERVED
from aura.environments.procedural_world import ProceduralWorld


def test_doorway_partition_and_equal_teacher_budget():
    assert set(TRAIN).isdisjoint(RESERVED)
    assert set(TRAIN)|set(RESERVED)==set(range(1,8))
    for c in range(1,8):
        env=DoorwayWorld(c)
        assert env.door==(4,c)
        assert {j for j in range(1,8) if (4,j) not in env.walls}=={c}
        ep=demonstration(c,42)
        assert len(ep)==13 and ep[-1].terminated
        assert all(np.array_equal(a.next_observation,b.observation) for a,b in zip(ep,ep[1:]))


def test_center_matches_original_observations():
    a=DoorwayWorld(4,noise=.05); b=ProceduralWorld('key',size=9,max_steps=60,noise=.05)
    assert np.array_equal(a.reset(seed=12)[0],b.reset(seed=12)[0])
    for action in [1,1,3,3,3,4,1,1,1,1,3,3,3]:
        x,y=a.step(action),b.step(action)
        assert np.array_equal(x[0],y[0]) and x[1:]==y[1:]
    with pytest.raises(ValueError): DoorwayWorld(0)
