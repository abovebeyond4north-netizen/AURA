import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.hybrid_routes import choose_route, HybridRouteCamera
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from aura.exploration.efficient_routes import EfficientRouteCamera


def make_view(walls):
    image = np.broadcast_to(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(3,7,7)).copy()
    image.reshape(3,-1)[:,:walls] = np.array(COLORS['wall'],dtype=np.uint8)[:,None]
    return image


def test_selection_uses_observed_wall_support():
    assert choose_route(make_view(11)) == 'early'
    assert choose_route(make_view(10)) == 'survey'


def test_invalid_pixels_are_not_wall_support():
    image = make_view(11)
    image[:,0,0] = (255,1,254)
    assert choose_route(image) == 'survey'


@pytest.mark.parametrize('walls,expected',[(11,'early'),(10,'survey')])
def test_dispatch_without_extra_interactions(monkeypatch,walls,expected):
    c = HybridRouteCamera(make_view(walls),lambda _:pytest.fail('unexpected step'))
    monkeypatch.setattr(AdaptiveRouteCamera,'prepare',lambda self,arm:'survey')
    monkeypatch.setattr(EfficientRouteCamera,'prepare',lambda self,arm:'early')
    assert c.prepare(2) == expected
    with pytest.raises(ValueError): c.prepare(1)


def test_motion_contract_inherited():
    assert HybridRouteCamera.act is AdaptiveRouteCamera.act
