import pytest
import torch
from aura.agent import AuraOmega
from aura.config import Config
from aura.memory.evidence_ledger import EvidenceLedger
from aura.memory.episodic import EpisodicMemory
from aura.meta_learning.resource_allocator import Budget
from aura.planning.distillation import RecurrentImitation
from aura.planning.online_distillation import OnlineDistillation
from test_distillation import episode


def test_scheduled_candidates_budget_evidence_and_no_activation(tmp_path):
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    config=Config(steps=100, imitation_interval=2, imitation_updates=4,
                  imitation_max_candidates=2, imitation_eval_episodes=1, max_episode_steps=5)
    budget=Budget(updates=8,candidates=2,imitation_eval_interactions=20)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    try:
        for i in range(4):
            for e in episode(i,i): memory.add(e)
            state=torch.random.get_rng_state().clone()
            scheduler.observe(episode(i,i)[-1],step=(i+1)*3)
            assert torch.equal(state,torch.random.get_rng_state())
        assert len(scheduler.reports)==2
        assert budget.used['updates']==8 and budget.used['candidates']==2
        for report in scheduler.reports:
            assert report['promotion']=='not_evaluated_by_protected_verifier'
            assert set(report['training_seeds']).isdisjoint(report['validation_seeds'])
            assert set(report['evaluation_seeds']).isdisjoint(report['training_seeds']+report['validation_seeds'])
            assert (tmp_path/report['checkpoint']).is_file()
            RecurrentImitation.load(tmp_path/report['checkpoint'])
        assert budget.used['imitation_eval_interactions']==sum(r['evaluation_interactions'] for r in scheduler.reports)
        assert ledger.verify() and len(ledger.records('imitation_candidate'))==2
        for e in episode(4,4): memory.add(e)
        scheduler.observe(episode(4,4)[-1],step=15)
        assert len(scheduler.reports)==2
    finally: ledger.close()


def test_no_candidate_for_failed_episode_or_exhausted_budget(tmp_path):
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    config=Config(imitation_interval=1,imitation_updates=3,imitation_max_candidates=2,imitation_eval_episodes=1,max_episode_steps=5)
    budget=Budget(updates=2,candidates=2,imitation_eval_interactions=20)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    try:
        for i in range(3):
            ep=episode(i,i,win=i!=0)
            for e in ep:memory.add(e)
            scheduler.observe(ep[-1],step=3*(i+1))
        assert not scheduler.reports and not list(tmp_path.iterdir())
        assert budget.used['updates']==0 and budget.used['candidates']==0
    finally:ledger.close()


def test_default_off_and_real_agent_boundary_integration(tmp_path):
    config=Config(seed=1082,steps=180,max_episode_steps=80,curiosity_policy='reward_routes',
                  warmup=8,batch_size=8,latent_dim=8,hidden_dim=16,ensemble_size=2,
                  temporal_updates_per_episode=0,imitation_interval=2,imitation_updates=4,
                  imitation_max_candidates=1,imitation_eval_episodes=1)
    agent=AuraOmega(config,tmp_path)
    try:
        result=agent.train(condition='curiosity',task='key',log_every=180)
        assert len(result['imitation_candidates'])==1
        assert result['resource_metrics']['imitation_evaluation_interactions']>0
        assert result['resource_metrics']['imitation_optimizer_updates']==4
        assert agent.ledger.verify()
        assert agent.explorer is agent.reward_routes
    finally:agent.close()
    assert Config().imitation_interval==0


def test_successor_uses_parent_weights_and_new_seed_groups(tmp_path, monkeypatch):
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    config=Config(imitation_interval=2,imitation_updates=4,imitation_max_candidates=3,
                  imitation_eval_episodes=1,max_episode_steps=5,imitation_inherit=True)
    budget=Budget(updates=12,candidates=3,imitation_eval_interactions=30)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    from aura.planning import online_distillation as module
    original=module.fit_imitation
    starting=[]
    def traced(model, *args, **kwargs):
        starting.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        return original(model,*args,**kwargs)
    monkeypatch.setattr(module,'fit_imitation',traced)
    try:
        for i,seed in enumerate((1,2,1,2,3,4,5,6)):
            ep=episode(i,seed)
            for e in ep: memory.add(e)
            scheduler.observe(ep[-1],step=(i+1)*3)
            if i==3: assert len(scheduler.reports)==1
        assert len(scheduler.reports)==3
        known=set()
        for i,report in enumerate(scheduler.reports):
            seeds=set(report['training_seeds']+report['validation_seeds'])
            assert not known.intersection(seeds)
            known.update(seeds)
            assert report['parent_candidate']==(i if i else None)
            assert report['promotion']=='not_evaluated_by_protected_verifier'
            assert [r['seed'] for r in report['before']]==[r['seed'] for r in report['after']]
            if i:
                parent=scheduler.reports[i-1]
                assert report['parent_checkpoint_sha256']==parent['checkpoint_sha256']
                model=RecurrentImitation.load(tmp_path/parent['checkpoint'])
                assert all(torch.equal(v,starting[i][k]) for k,v in model.state_dict().items())
        assert budget.used['candidates']==3 and budget.used['updates']==12
        assert ledger.verify()
    finally: ledger.close()


def test_successor_rejects_modified_parent_before_spending(tmp_path):
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    config=Config(imitation_interval=2,imitation_updates=3,imitation_max_candidates=2,
                  imitation_eval_episodes=1,max_episode_steps=5,imitation_inherit=True)
    budget=Budget(updates=6,candidates=2,imitation_eval_interactions=20)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    try:
        for i in range(4):
            ep=episode(i,i)
            for e in ep: memory.add(e)
            if i==3:
                before=budget.used.copy()
                with pytest.raises(ValueError,match='Parent checkpoint digest'):
                    scheduler.observe(ep[-1],step=(i+1)*3)
                assert budget.used==before and len(scheduler.reports)==1
            else: scheduler.observe(ep[-1],step=(i+1)*3)
            if i==1:
                (tmp_path/scheduler.reports[0]['checkpoint']).write_bytes(b'corrupt')
        assert ledger.verify()
    finally: ledger.close()


@pytest.mark.parametrize("rehearsal", [0, 2])
def test_lineage_diagnostic_replays_checkpoints_and_rejects_tamper(tmp_path, rehearsal):
    import json
    from experiments.imitation_lineage_study import compare
    ledger=EvidenceLedger(tmp_path/'evidence.sqlite'); memory=EpisodicMemory(ledger)
    config=Config(imitation_interval=2,imitation_updates=3,imitation_max_candidates=2,
                  imitation_eval_episodes=1,max_episode_steps=5,imitation_inherit=True,
                  imitation_rehearsal_episodes=rehearsal)
    budget=Budget(updates=6,candidates=2,imitation_eval_interactions=20)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    try:
        for i in range(4):
            ep=episode(i,i)
            for e in ep: memory.add(e)
            scheduler.observe(ep[-1],step=(i+1)*3)
    finally: ledger.close()
    (tmp_path/'study.json').write_text(json.dumps({'config':config.model_dump(),
                       'result':{'imitation_candidates':scheduler.reports}}))
    result=compare(tmp_path,episodes=1)
    generations=result['generations']
    assert len(generations)==2 and all(g['restored_shadow_rows_equal'] for g in generations)
    assert [r['seed'] for r in generations[0]['rows']]==[r['seed'] for r in generations[1]['rows']]
    assert set(r['seed'] for r in generations[0]['rows']).isdisjoint(range(4))
    assert result['diagnostic_interactions']==sum(r['steps'] for g in generations for r in g['rows'])
    (tmp_path/scheduler.reports[-1]['checkpoint']).write_bytes(b'bad')
    with pytest.raises(ValueError,match='Checkpoint digest mismatch'):compare(tmp_path,episodes=1)


def test_rehearsal_uses_only_ancestral_training_evidence(tmp_path):
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    config=Config(imitation_interval=2,imitation_updates=3,imitation_max_candidates=3,
                  imitation_eval_episodes=1,max_episode_steps=5,imitation_inherit=True,
                  imitation_rehearsal_episodes=2)
    budget=Budget(updates=9,candidates=3,imitation_eval_interactions=30)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    try:
        for i in range(6):
            ep=episode(i,i)
            for e in ep: memory.add(e)
            scheduler.observe(ep[-1],step=(i+1)*3)
        assert len(scheduler.reports)==3
        trained=set(); held=set(); fresh=set()
        for report in scheduler.reports:
            prior={tuple(ids) for ids in report['rehearsal_evidence']}
            new={tuple(ids) for ids in report['fresh_training_evidence']}
            val={tuple(ids) for ids in report['validation_evidence']}
            assert prior <= trained and not (prior & held)
            assert not ((new|val)&fresh)
            assert new|prior=={tuple(ids) for ids in report['training_evidence']}
            assert len(prior)<=2
            assert not (set(report['training_seeds']) & set(report['validation_seeds']))
            if trained: assert prior
            trained |= new; held |= val; fresh |= new|val
        assert budget.used['updates']==9 and ledger.verify()
    finally: ledger.close()


def test_budgeted_rehearsal_integrated_cost_and_presented_provenance(tmp_path):
    from aura.planning.online_distillation import ancestral_rehearsal
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    config=Config(imitation_interval=3, imitation_updates=4, imitation_max_candidates=2,
                  imitation_eval_episodes=1, max_episode_steps=5, imitation_inherit=True,
                  imitation_rehearsal_episodes=2, imitation_budgeted_rehearsal=True)
    budget=Budget(updates=8,candidates=2,imitation_eval_interactions=20)
    scheduler=OnlineDistillation(config,ledger,budget,tmp_path)
    try:
        for i in range(6):
            ep=episode(i,i)
            for e in ep: memory.add(e)
            scheduler.observe(ep[-1],step=(i+1)*3)
        assert len(scheduler.reports)==2
        a,b=scheduler.reports
        assert a['training_transition_presentations']==b['training_transition_presentations']==24
        assert b['training_padded_step_presentations']==24
        assert b['rehearsal_transition_presentations']==12
        assert set(map(tuple,b['rehearsal_evidence'])) <= set(map(tuple,a['presented_training_evidence']))
        assert b['presented_training_evidence']==b['training_evidence']
        assert ledger.verify() and budget.used['updates']==8
        # An available but never presented episode may not be rehearsed later.
        unused=dict(a, presented_training_evidence=[])
        from aura.planning.distillation import rewarded_episodes
        assert ancestral_rehearsal(rewarded_episodes(ledger),[unused],2)==[]
    finally: ledger.close()
