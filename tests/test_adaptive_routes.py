from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from aura.exploration.observable_routes import ObservableRouteCamera
from experiments.goal_acquisition import trial


def test_adaptive_route_cost_prefers_distinguishable_edges():
    c=object.__new__(AdaptiveRouteCamera)
    c.position=(0,0)
    c.map={(r,k):'wall' if r==2 else 'empty' for r in range(-5,6) for k in range(-5,6)}
    assert c.route_cost((3,)) > c.route_cost((1,))
    assert c.route_cost(()) == 0


def test_clean_large_layouts_and_reversal():
    for size,col in [(13,4),(15,7)]:
        for rev in (False,True):
            args=(86100,size,col,rev,0,'frontier')
            assert not trial(*args,ObservableRouteCamera)['success']
            row=trial(*args,AdaptiveRouteCamera)
            assert row['success'] and len(row['actions']) <= 65


def test_motion_verification_unchanged():
    assert AdaptiveRouteCamera.act is ObservableRouteCamera.act
