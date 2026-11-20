"""Old-task retention and new-task adaptation must be measured separately."""
from experiments.evaluate_continual import run_one

def test_replay_helps_retention_on_contextual_rule_switch():
    control=run_one(seed=0,initial_updates=80,adapt_updates=80,replay=False)
    rehearsal=run_one(seed=0,initial_updates=80,adapt_updates=80,replay=True)
    assert control['old_mse_after_stage2'] > control['old_mse_after_stage1']+.01
    assert rehearsal['old_mse_after_stage2'] < control['old_mse_after_stage2']*.5
    # Both models retain viable adaptation to the new observable rule.
    assert rehearsal['new_mse_after_stage2']<.03
    assert control['new_mse_after_stage2']<.03
    assert control['updates']==rehearsal['updates']==160
    assert control['rehearsal_buffer_bytes']==0
    assert rehearsal['rehearsal_buffer_bytes']>0
    assert 0<=rehearsal['plasticity']['inactive_fraction']<=1
    assert 0<=control['plasticity']['inactive_fraction']<=1
