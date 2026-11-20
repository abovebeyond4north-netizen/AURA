import numpy as np
import pytest
from aura.world_model.dynamics import WorldModel

def test_recurrent_imagination_not_observation():
    model=WorldModel(8,16,2)
    x=np.random.default_rng(3).integers(0,256,(1,3,7,7),dtype=np.uint8)
    z=model.imagine(x,[0,1,4])
    assert len(z)==3 and z[-1].shape==(1,8)

def test_checkpoint_roundtrip_and_corrupt_rejected(tmp_path):
    m=WorldModel(8,16,2)
    p=tmp_path/'safe.pt';m.save(p,{'seed':42})
    n=WorldModel(8,16,2)
    assert n.restore(p)=={'seed':42}
    assert all((a==b).all() for a,b in zip(m.parameters(),n.parameters()))
    p.write_bytes(b'not a safe checkpoint')
    with pytest.raises(Exception):n.restore(p)

def test_nonfinite_gradient_rejected():
    m=WorldModel(8,16,2)
    x=np.full((4,3,7,7),255,dtype=np.uint8)
    y=x.copy(); y=y.astype(float);y[0,0,0,0]=np.nan
    with pytest.raises(FloatingPointError): m.train_batch((x,np.zeros(4,dtype=int),y))
