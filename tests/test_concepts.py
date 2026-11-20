import numpy as np
from aura.cognition.concepts import ConceptDiscovery

def test_predictively_useful_prototypes_can_be_retained():
    rng=np.random.default_rng(0)
    a=rng.normal(-4,.2,(60,3));b=rng.normal(4,.2,(60,3))
    x=np.concatenate([a,b]);y=np.concatenate([np.zeros(60),np.ones(60)])[:,None]
    permutation=rng.permutation(len(x));x=x[permutation];y=y[permutation]
    c=ConceptDiscovery(penalty=.0001)
    result=c.fit(x,y)
    assert result['accepted'] and c.assign(x[:5]).shape==(5,)
