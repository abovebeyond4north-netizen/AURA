import numpy as np
from aura.exploration.goal_search import map_shift
from aura.environments.procedural_world import COLORS

def test_map_shift_unique_and_ambiguous():
    rng=np.random.default_rng(8)
    tiles={(r,c): ('wall' if rng.random()<.35 else 'empty') for r in range(-5,6) for c in range(-5,6)}
    image=np.array([[COLORS[tiles[r-3,c-2]] for c in range(7)] for r in range(7)],dtype=np.uint8).transpose(2,0,1)
    image[:,3,3]=COLORS['agent']
    assert map_shift(image,(0,0),tiles)==(0,1)
    assert map_shift(image,(0,0),{}) is None
    uniform={p:'empty' for p in tiles}
    image[:]=np.array(COLORS['empty'])[:,None,None]
    assert map_shift(image,(0,0),uniform) is None
