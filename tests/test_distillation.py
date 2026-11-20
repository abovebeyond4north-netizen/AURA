import numpy as np
import pytest
import torch
from aura.types import Experience
from aura.memory.evidence_ledger import EvidenceLedger
from aura.memory.episodic import EpisodicMemory
from aura.planning.distillation import rewarded_episodes,split_episodes,RecurrentImitation,fit_imitation


def episode(i,seed,win=True,broken=False):
    obs=[np.full((3,7,7),j*50,dtype=np.uint8) for j in range(4)]
    return tuple(Experience(obs[j],j,obs[j+1] if not broken or j!=0 else obs[3],1. if win and j==2 else -.01,
                 win and j==2,not win and j==2,str(i),f'{i}:{j}',seed=seed) for j in range(3))


def test_evidence_selection_and_seed_groups():
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    try:
        for ep in (episode(0,1),episode(1,1),episode(2,2),episode(3,3,False),episode(4,4)[:2]):
            for e in ep: memory.add(e)
        eligible=rewarded_episodes(ledger); assert len(eligible)==3
        a,b=split_episodes(eligible); assert {e[0].seed for e in a}.isdisjoint(e[0].seed for e in b)
        assert len(a)+len(b)==3
        for e in episode(5,5,broken=True): memory.add(e)
        with pytest.raises(ValueError,match='Inconsistent'): rewarded_episodes(ledger)
    finally: ledger.close()


def test_training_overlap_rejected_and_neural_roundtrip(tmp_path):
    torch.manual_seed(6); torch.set_num_threads(2)
    model=RecurrentImitation(16)
    with pytest.raises(ValueError,match='overlap'): fit_imitation(model,[episode(1,2)],[episode(2,2)],updates=1)
    result=fit_imitation(model,[episode(1,1),episode(2,2)],[episode(3,3)],updates=60)
    assert result['after']['validation_loss']<result['before']['validation_loss']
    assert result['after']['validation_accuracy']==1.
    policy=tmp_path/'imitation.pt'; model.save(policy); restored=RecurrentImitation.load(policy)
    obs=[e.observation for e in episode(3,3)]
    assert [model.choose(x) for x in obs]==[restored.choose(x) for x in obs]
    model.reset_episode(); assert [model.choose(x) for x in obs]==[0,1,2]


def test_invalid_ledger_and_optimizer_budget():
    ledger=EvidenceLedger(); memory=EpisodicMemory(ledger)
    try:
        for e in episode(0,1):memory.add(e)
        ledger.con.execute('DROP TRIGGER ledger_no_update')
        ledger.con.execute("UPDATE ledger SET record_hash='invalid' WHERE sequence=1")
        with pytest.raises(ValueError,match='Invalid evidence ledger'):rewarded_episodes(ledger)
    finally:ledger.close()
    with pytest.raises(ValueError,match='budget'):
        fit_imitation(RecurrentImitation(16),[episode(1,1)],[episode(2,2)],lr=float('nan'))


def test_budgeted_rehearsal_preserves_fresh_transition_and_padding_caps():
    from aura.planning.distillation import budgeted_episode_indices
    lengths = [3, 5, 7, 4, 6, 30]
    batches = [budgeted_episode_indices(lengths, 3, step) for step in range(12)]
    assert all(batch == budgeted_episode_indices(lengths, 3, step) for step, batch in enumerate(batches))
    for batch in batches:
        assert len(batch) == len(set(batch)) and any(i < 3 for i in batch)
        assert sum(lengths[i] for i in batch) <= 15
        assert len(batch) * max(lengths[i] for i in batch) <= 21
    assert set().union(*(set(batch) for batch in batches)) == {0, 1, 2, 3, 4}
    assert budgeted_episode_indices([3, 5, 7], 3, 0) == [0, 1, 2]


def test_budgeted_fit_accounts_for_actual_optimizer_presentations():
    training = [episode(i, i) for i in range(4)]
    result = fit_imitation(RecurrentImitation(16), training, [episode(5,5)],
                           updates=8, budgeted_fresh_count=2)
    assert result['training_transition_presentations'] == 48
    assert result['training_padded_step_presentations'] == 48
    assert sum(result['episode_presentations']) == 16
    assert all(n > 0 for n in result['episode_presentations'])
    assert result['rehearsal_transition_presentations'] == 24
    assert result['training_transitions'] == 12  # available data, not work performed
    assert result['fresh_transition_cap_per_update'] == 6
    with pytest.raises(ValueError, match='fresh'):
        fit_imitation(RecurrentImitation(16), training, [episode(5,5)], updates=1, budgeted_fresh_count=0)
