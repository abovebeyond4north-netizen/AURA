import pytest
from aura.curriculum.generator import Curriculum
from aura.meta_learning.controller import MetaController
from aura.meta_learning.resource_allocator import Budget

def test_curriculum_and_meta_selection():
    c=Curriculum(seed=3)
    for i in range(30): c.update('key',True)
    assert c.choose_goal() in c.tasks and c.choices[-1]['reason']
    m=MetaController(('EXPLORE','TRAIN'))
    assert m.select()=='EXPLORE';m.update('EXPLORE',1,.1)
    assert m.select()=='TRAIN';m.update('TRAIN',0,.1)
    assert m.select() in ('EXPLORE','TRAIN')

def test_budget_exhaustion():
    b=Budget(interactions=1)
    b.spend('interactions'); assert not b.remaining()
    with pytest.raises(RuntimeError):b.spend('interactions')
