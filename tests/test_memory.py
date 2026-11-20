import sqlite3
import numpy as np
import pytest
from aura.types import Source, Experience,Hypothesis
from aura.memory.evidence_ledger import EvidenceLedger
from aura.memory.replay import ExperienceReplay
from aura.memory.semantic import SemanticMemory
from aura.memory.episodic import EpisodicMemory

def sample():
    x=np.zeros((3,7,7),dtype=np.uint8)
    return Experience(x,1,x.copy(),-.01,False,False,'e0','id0')

def test_append_only_hash_chain_and_imagination_rejection():
    ledger=EvidenceLedger();ledger.append('a','experience',Source.OBSERVED,{'ok':True})
    assert ledger.verify()
    with pytest.raises(sqlite3.DatabaseError): ledger.con.execute("UPDATE ledger SET kind='fake'")
    with pytest.raises(sqlite3.DatabaseError): ledger.con.execute('DELETE FROM ledger')
    with pytest.raises(ValueError): ledger.append('b','experience',Source.IMAGINED,{'fake':True})
    ledger.close()

def test_replay_empty_and_source_contract():
    replay=ExperienceReplay();assert len(replay)==0
    with pytest.raises(ValueError): replay.sample(1)
    replay.add(sample());assert replay.sample(1)[0].shape==(1,3,7,7)
    with pytest.raises(ValueError): Experience(sample().observation,1,sample().next_observation,0,False,False,'a','b',source=Source.IMAGINED)

def test_semantic_revisions_preserve_contradiction():
    ledger=EvidenceLedger()
    s=SemanticMemory(ledger)
    h=Hypothesis('h0',{'effect':True},evidence_ids=['a'])
    s.revise(h);h.counterevidence_ids.append('b');h.confidence=.3;s.revise(h)
    records=ledger.records('hypothesis')
    assert len(records)==2 and records[0]['payload']['confidence']==.5
    assert records[1]['payload']['counterevidence_ids']==['b'] and ledger.verify()


def test_lossless_observation_persistence():
    ledger=EvidenceLedger();mem=EpisodicMemory(ledger)
    original=sample();mem.add(original)
    restored=mem.reconstruct(ledger.records('experience')[0])
    assert np.array_equal(restored.observation,original.observation)
    assert restored.action==original.action and ledger.verify()
