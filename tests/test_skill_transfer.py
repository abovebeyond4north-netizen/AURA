import pytest
from aura.environments.skill_workbench import SkillWorkbench
from aura.skills.online_options import OnlineOptionLearner,run_skill_transfer

def test_visual_environment_never_exposes_mechanism():
    env=SkillWorkbench('composition',mechanism_seed=117)
    obs,meta=env.reset(seed=1)
    assert obs.shape==(3,7,7) and not meta
    with pytest.raises(ValueError): env.step(5)
    assert 'mapping' not in meta

def test_positive_evidence_required_for_skill():
    learner=OnlineOptionLearner(seed=3)
    obs,_=SkillWorkbench().reset(seed=4)
    for i in range(5): learner.observe(obs,0,0.,f'e{i}')
    assert not learner.options()
    for i in range(3): learner.observe(obs,3,1.,f'good{i}')
    assert len(learner.options())==1 and learner.options()[0].action==3

def test_unseen_composition_benefits_from_learned_options():
    reports=[run_skill_transfer(seed=s,train_episodes=80,test_episodes=60) for s in range(5)]
    opts=sum(r['trained_vs_random']['options']['wins'] for r in reports)
    rand=sum(r['trained_vs_random']['random']['wins'] for r in reports)
    assert all(r['positive_reward_options']>=2 for r in reports)
    assert opts>=250 and opts>rand+200, (opts,rand)
