import pytest
from aura.environments.procedural_world import ProceduralWorld
from protected_verifier.solvability import evaluator_solution

@pytest.mark.parametrize('task',('key','switch','hidden','altered','noise'))
def test_every_primary_family_has_evaluator_only_solution(task):
    world=ProceduralWorld(task,max_steps=45,diagnostics=True)
    world.reset(seed=3)
    path=evaluator_solution(world)
    assert path is not None and len(path)<=45,task
    _,meta=world.reset(seed=3)
    assert 'rule' not in meta
    outcome=None
    for a in path: outcome=world.step(a)
    assert outcome[2] is True

def test_learner_cannot_request_privileged_plan():
    world=ProceduralWorld('hidden',diagnostics=False)
    with pytest.raises(PermissionError):evaluator_solution(world)
