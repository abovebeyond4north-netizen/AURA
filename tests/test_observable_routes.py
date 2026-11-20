from aura.exploration.observable_routes import ObservableRouteCamera, stationary_contrasts
from aura.exploration.goal_search import PassageRecoveryCamera
from experiments.goal_acquisition import trial


def test_stationary_contrast_does_not_invent_observations():
    uniform = {(r,c): 'empty' for r in range(-5,6) for c in range(-5,6)}
    assert stationary_contrasts(uniform, (0,0), (0,1)) == 0
    assert stationary_contrasts({}, (0,0), (0,1)) == 0
    wall = {p: 'wall' if p[0] == 2 else t for p,t in uniform.items()}
    assert stationary_contrasts(wall, (0,0), (0,1)) == 0
    assert stationary_contrasts(wall, (0,0), (1,0)) > 0


def test_route_detours_while_excluding_blocked_tiles():
    camera = object.__new__(ObservableRouteCamera)
    camera.position = (0,0)
    camera.landmarks = {'key': (0,2), 'door': (1,2)}
    camera.map = {(r,c): 'empty' for r in range(2) for c in range(4)}
    camera.map[(0,2)] = 'key'; camera.map[(1,2)] = 'door'
    paths = camera.paths()
    assert paths[(1,1)] == (1,3)
    assert (0,2) not in paths and (1,2) not in paths and (0,3) not in paths


def test_original_alias_failure_resolved_clean():
    args = (85100,13,6,False,0,'frontier')
    assert not trial(*args,PassageRecoveryCamera)['success']
    assert trial(*args,ObservableRouteCamera)['success']
    assert ObservableRouteCamera.act is PassageRecoveryCamera.act
