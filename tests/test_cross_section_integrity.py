"""Cross-section regressions for task-local evidence, skill provenance and credit."""
import numpy as np
import pytest
from aura.agent import AuraOmega
from aura.config import Config
from aura.skills.discovery import SkillDiscovery
from aura.types import Experience
from aura.exploration.visual_frontier import VisualFrontierExplorer
from aura.meta_learning.controller import MetaController


def test_curriculum_uses_current_task_development_and_fresh_credit(tmp_path, monkeypatch):
    agent=AuraOmega(Config(steps=48,max_episode_steps=12,warmup=4,batch_size=4,latent_dim=8,hidden_dim=16,ensemble_size=2,temporal_updates_per_episode=0,log_dir=str(tmp_path)))
    tasks=iter(['key','switch','key','switch','key'])
    monkeypatch.setattr(agent.curriculum,'choose_goal',lambda:next(tasks))
    calls=[]
    original=agent.meta.update
    def capture(mode,gain,cost):
        calls.append(gain)
        return original(mode,gain,cost)
    monkeypatch.setattr(agent.meta,'update',capture)
    try:
        report=agent.train(condition='omega')
        banks=report['development_validation_by_task']
        assert set(banks)=={'key','switch'}
        assert banks['key']['sha256']!=banks['switch']['sha256']
        assert report['resource_metrics']['development_interactions']==16
        assert len(agent.ledger.records('development_validation'))==2
        measurements=agent.ledger.records('development_gain')
        assert {m['payload']['task'] for m in measurements}=={'key','switch'}
        for m in measurements:
            p=m['payload']
            assert p['bank_sha256']==banks[p['task']]['sha256']
            assert p['gain']==pytest.approx(p['before']-p['after'])
        assert sum(g is not None for g in calls)<=len(measurements)
        assert any(g is None for g in calls)
        assert report['ledger_valid']
    finally:agent.close()


def episode(tag, actions):
    a=np.zeros((3,7,7),dtype=np.uint8)
    return [Experience(a,v,a,1. if i==len(actions)-1 else -.01,i==len(actions)-1,False,tag,f'{tag}:{i}') for i,v in enumerate(actions)]


def test_skill_support_is_unique_episodes_and_never_reuses_ids():
    d=SkillDiscovery(min_occurrences=2,max_length=2)
    e=episode('a',[1,2,1,2])
    assert not d.observe_episode(e,True,'key')
    assert not d.observe_episode(e,True,'key')
    found=d.observe_episode(episode('b',[1,2,1,2]),True,'key')
    assert found
    assert {r.split(':')[0] for r in d.skill_library[found[0]].skill.evidence_ids}=={'a','b'}
    old_ids=set(found)
    d.skill_library.clear()
    d.observe_episode(episode('c',[3,4]),True,'key')
    new=d.observe_episode(episode('d',[3,4]),True,'key')
    assert new and not old_ids.intersection(new)


def test_meta_does_not_turn_missing_measurements_into_observations():
    m=MetaController(('A','B'))
    m.update('A',None,1.)
    m.update('A',-.2,1.)
    assert m.counts[0]==2 and m.gain_counts[0]==1 and m.gains[0]==pytest.approx(-.2)
    with pytest.raises(ValueError):m.update('A',float('nan'),1.)
    with pytest.raises(ValueError):m.update('A',1.,-1.)


def test_frontier_reopens_previously_blocked_attempt_after_scene_change(monkeypatch):
    p=VisualFrontierExplorer(seed=7,epsilon=0)
    x=np.zeros((3,7,7),dtype=np.uint8)
    p.track_action(x,1); p.observe(x,1,x)
    assert (p.pose,1) in p.blocked and p.attempts[(p.pose,1)]==1
    monkeypatch.setattr(p.predictor,'_stable_effect',lambda *_:.5)
    p.track_action(x,4); p.observe(x,4,x)
    assert (p.pose,1) not in p.blocked
    assert p.attempts[(p.pose,1)]==0
    assert p._has_frontier(p.pose)


def test_online_visual_effect_is_not_labelled_randomized_causation(tmp_path):
    agent=AuraOmega(Config(steps=30,log_dir=str(tmp_path)))
    try:
        agent.actor.select=lambda *args:4
        agent.train(condition='fixed',task='key')
        hypotheses=agent.ledger.records('hypothesis')
        assert hypotheses
        assert all(h['payload']['expression']['kind']=='observational' for h in hypotheses)
    finally:agent.close()


def test_each_repeatedly_selected_meta_mode_receives_gain_samples(tmp_path):
    agent=AuraOmega(Config(steps=120,max_episode_steps=30,warmup=4,batch_size=4,latent_dim=8,hidden_dim=16,ensemble_size=2,temporal_updates_per_episode=0,log_dir=str(tmp_path)))
    try:
        result=agent.train(task='key',condition='omega')
        m=result['meta_selection_metrics']
        selected=[k for k,v in m['selections'].items() if v>=10]
        assert len(selected)>=4
        assert all(m['measured_gains'][k]>=1 for k in selected)
    finally:agent.close()


def test_curriculum_covers_unattempted_families_before_repeating():
    from aura.curriculum.generator import Curriculum
    c=Curriculum(seed=42); chosen=[]
    for _ in c.tasks:
        t=c.choose_goal(); chosen.append(t); c.update(t,False)
    assert set(chosen)==set(c.tasks)
    assert all('unattempted' in r['reason'] for r in c.choices)
