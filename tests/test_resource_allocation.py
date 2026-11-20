import pytest
from aura.meta_learning.resource_allocator import Budget,ValueOfComputationAllocator
from aura.config import Config
from aura.agent import AuraOmega

def test_updates_can_run_out_without_training_crash(tmp_path):
    a=AuraOmega(Config(steps=18,batch_size=6,warmup=6,latent_dim=8,
                        hidden_dim=16,ensemble_size=2,log_dir=str(tmp_path)))
    a.budget=Budget(interactions=18,updates=0,seconds=60)
    result=a.train(condition='fixed')
    assert result['steps']==18 and result['temporal_training_steps']==0
    assert a.budget.used['updates']==0
    a.close()

def test_value_of_computation_rejects_high_cost():
    allocator=ValueOfComputationAllocator(('A','B'),bonus=0)
    assert allocator.select()=='A'
    for _ in range(3):
        allocator.update('A',validated_gain=.1,measured_cost=.4)
        allocator.update('B',validated_gain=.2,measured_cost=.01)
    assert allocator.select()=='B'
    for _ in range(3):allocator.update('B',validated_gain=0.,measured_cost=1.)
    assert allocator.select() is None
    with pytest.raises(ValueError):allocator.update('A',0.,float('nan'))
