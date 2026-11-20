import numpy as np
import pytest
from aura.types import Experience
from aura.skills.discovery import SkillDiscovery
from aura.skills.composition import compose
from aura.environments.procedural_world import create_environment

def eps(actions,episode_id="e"):
    x=np.zeros((3,7,7),dtype=np.uint8)
    return [Experience(x,a,x,0,False,False,episode_id,f'{episode_id}:r{i}') for i,a in enumerate(actions)]

def test_skill_mined_repeated_success_and_composed():
    d=SkillDiscovery(min_occurrences=2,max_length=3)
    assert not d.observe_episode(eps([1,2,3]),False,'key')
    d.observe_episode(eps([1,2,3]),True,'key')
    found=d.observe_episode(eps([1,2,3],'e2'),True,'key')
    assert found
    a=d.skill_library[found[0]]
    b=compose(a,a,'key')
    assert len(b.actions)==len(a.actions)*2
    with pytest.raises(ValueError):compose(a,a,'switch')
    env=create_environment('key',max_steps=15)
    _,_,_=b.execute(env,'key')
    assert b.skill.failures+b.skill.successes==1
