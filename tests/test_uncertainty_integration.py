"""Uncertainty shift fit is produced from actual training data, not test seeds."""
from aura.agent import AuraOmega
from aura.config import Config

def test_online_observation_shift_calibration_is_logged_and_consumed(tmp_path):
    agent=AuraOmega(Config(steps=65,batch_size=6,warmup=6,latent_dim=8,
                           hidden_dim=16,ensemble_size=2,log_dir=str(tmp_path)))
    result=agent.train(condition='omega')
    assert result['ledger_valid'] and agent.uncertainty.shift_is_calibrated
    fits=agent.ledger.records('calibration')
    assert len(fits)==1
    assert len(fits[0]['payload']['evidence_ids'])==60
    assert agent.uncertainty.shift_report(agent.replay.data[0].observation)['method'].startswith('frozen')
    agent.close()
