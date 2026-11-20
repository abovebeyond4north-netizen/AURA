import numpy as np
from aura.exploration.goal_search import palette_shift, PaletteGoalCamera
from aura.environments.procedural_world import COLORS
from experiments.goal_acquisition import trial

def test_masked_translation():
    rng=np.random.default_rng(31)
    colors=np.array(list(COLORS.values()),dtype=np.uint8)
    scene=colors[rng.integers(len(colors),size=(7,8))].transpose(2,0,1)
    a,b=scene[:,:,:7].copy(),scene[:,:,1:].copy()
    a[:,1,1]=b[:,2,2]=[1,2,3]
    assert palette_shift(a,b)==(0,1)

def test_abstention():
    a=np.full((3,7,7),110,dtype=np.uint8)
    assert palette_shift(a,a) is None
    a[:,:,:]=1
    assert palette_shift(a,a) is None

def test_clean_geometry():
    assert trial(72100,11,4,True,0,'frontier',PaletteGoalCamera)['success']
