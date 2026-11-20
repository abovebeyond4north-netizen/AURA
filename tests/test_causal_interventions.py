from aura.environments.causal_workbench import CausalWorkbench
from aura.cognition.intervention_study import run_causal_study

def test_confounding_creates_false_passive_association():
    passive=run_causal_study(CausalWorkbench('confounded',seed=5),400,seed=5,randomized=False)
    assert passive['result']=='NOT_IDENTIFIED'
    assert passive['estimate']['association']>.5

def test_randomized_control_recovers_no_causal_effect():
    study=run_causal_study(CausalWorkbench('confounded',seed=5),400,seed=5)
    assert study['result']=='NEGLIGIBLE_EFFECT_SUPPORTED', study
    assert abs(study['estimate']['effect'])<.15

def test_randomized_control_detects_real_effect():
    study=run_causal_study(CausalWorkbench('effective',seed=9),400,seed=9)
    assert study['result']=='POSITIVE_EFFECT_SUPPORTED', study
    assert study['estimate']['effect']>.65
