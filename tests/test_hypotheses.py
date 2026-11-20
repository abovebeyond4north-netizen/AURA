from aura.cognition.hypotheses import HypothesisEngine
from aura.cognition.causal_reasoning import compare_randomized_interventions
from aura.exploration.experiment_selection import ExperimentSelector

def test_hypothesis_revision_and_inconclusive():
    e=HypothesisEngine()
    assert e.classify('x',4)=='INCONCLUSIVE'
    for i in range(8):e.update('x',4,True,f'e{i}')
    h=e.form('x',4);assert h.confidence>.7 and len(h.evidence_ids)==8
    e.update('x',4,False,'contra')
    assert h.confidence>e.form('x',4).confidence
    assert e.form('x',4).counterevidence_ids==['contra']
    assert ExperimentSelector.choose([h]) in range(5)

def test_causal_inference_needs_interventions_and_sufficient_samples():
    assert compare_randomized_interventions([True,False],[1,0])['identified'] is False
    result=compare_randomized_interventions([True]*10+[False]*10,[1]*10+[0]*10)
    assert result['identified'] is True and result['effect']==1
