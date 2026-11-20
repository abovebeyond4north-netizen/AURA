import numpy as np
import torch
from aura.perception.encoder import ObservationEncoder
from aura.world_model.temporal_heads import TemporalHeads

def controlled_trajectory(n=12,seed=19):
    rng=np.random.default_rng(seed)
    # A controllable visual dynamics process with deterministic action effects.
    state=np.zeros((3,7,7),dtype=np.uint8)
    states=[state.copy()]; actions=[]; ends=[]
    for i in range(n):
        a=int(rng.integers(5));actions.append(a)
        state=state.copy()
        state[a%3,:,:]=np.clip(state[a%3,:,:].astype(int)+20,0,255)
        states.append(state.copy());ends.append(i==n-1)
    return np.stack(states),actions,ends

def test_multi_horizon_heads_train_and_improve():
    torch.manual_seed(13)
    m=TemporalHeads(ObservationEncoder(8),8,(1,2,4),hidden_dim=24,lr=.003)
    traj=controlled_trajectory()
    heldout=controlled_trajectory(seed=41)
    before=m.evaluate([heldout])
    out=m.train_trajectories([traj],updates=120)
    after=m.evaluate([heldout])
    assert out['updates']==120
    assert all(after[h]<before[h] for h in (1,2,4)), (before,after)

def test_windows_do_not_cross_episode_termination():
    m=TemporalHeads(ObservationEncoder(8),8,(1,2,4))
    traj=controlled_trajectory(6)
    obs,a,e=traj
    e[1]=True
    assert len(list(m._windows((obs,a,e),4)))==1
    assert m.evaluate([traj])[1] is not None
