"""Small procedural partially observed gridworld with image observations.

A Gymnasium Env if installed, or a deliberately small offline-compatible API fallback.
Hidden task mechanisms are never included in observations; diagnostic info is
only available to a separately invoked evaluator through `diagnostic_state`.
"""
from __future__ import annotations
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    class _Env: pass
    class _Box:
        def __init__(self, low, high, shape, dtype): self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
        def contains(self, x):
            a = np.asarray(x)
            return a.shape == self.shape and (a >= self.low).all() and (a <= self.high).all()
    class _Discrete:
        def __init__(self, n): self.n = n
        def contains(self, x): return isinstance(x, (int, np.integer)) and 0 <= x < self.n
    class gym: Env = _Env
    class spaces: Box = _Box; Discrete = _Discrete

TASKS = ('key', 'switch', 'hidden', 'altered', 'noise')
# Up/down/left/right/interact; no hidden goal disclosure.
MOVES = ((-1,0), (1,0), (0,-1), (0,1))
COLORS = {
    'empty': (8,8,12), 'wall': (110,110,110), 'agent': (255,240,120),
    'key': (255,210,0), 'switch': (0,200,255), 'door': (220,50,50),
    'goal': (40,225,80), 'crate': (160,85,40), 'unknown': (0,0,0)
}

class ProceduralWorld(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}
    def __init__(self, task: str = 'key', size: int = 9, max_steps: int = 50,
                 noise: float = 0, partial: bool = True, diagnostics: bool = False,
                 door_column: int | None = None, reverse_after_key: bool = False,
                 render_mode: str | None = 'rgb_array'):
        if task not in TASKS: raise ValueError(f'Unknown task: {task}')
        if render_mode not in (None, 'rgb_array'): raise ValueError('Unsupported render mode')
        self.render_mode = render_mode
        if size < 7 or size%2==0: raise ValueError('Grid size must be odd and at least 7')
        self.task, self.size, self.max_steps, self.noise, self.partial = task, size, max_steps, noise, partial
        if door_column is not None and (type(door_column) is not int or not 1 <= door_column < size-1):
            raise ValueError('Door column must be inside the boundary')
        self.door_column = size//2 if door_column is None else door_column
        self.reverse_after_key = bool(reverse_after_key)
        self._diagnostics = diagnostics
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 7, 7), dtype=np.uint8)
        self._rng = np.random.default_rng(0)
        self.steps = 0
        self.reset(seed=0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if HAS_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None: self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.agent = (1, 1)
        self.goal = (self.size-2, self.size-2)
        self.key = (2, 1)
        self.switch = (1, 2)
        self.door = (self.size//2, self.door_column)
        self.crate = (2, self.size-3)
        self.has_key = False
        self.door_open = False
        self.switch_on = False
        self.walls = {(0,i) for i in range(self.size)} | {(self.size-1,i) for i in range(self.size)}
        self.walls |= {(i,0) for i in range(self.size)} | {(i,self.size-1) for i in range(self.size)}
        # Solid interior barrier: reaching the goal requires opening the door.
        row = self.size//2
        self.walls |= {(row, col) for col in range(1,self.size-1) if col != self.door_column}
        self.walls.discard(self.door)
        self._rule = int(self._rng.integers(0,2)) if self.task == 'hidden' else 0
        observation = self._observe()
        self._last_observation = observation.copy()
        return observation, {'task_family': self.task}  # no hidden rule

    def _tile(self, p: tuple[int,int]) -> str:
        if p == self.agent: return 'agent'
        if p in self.walls: return 'wall'
        if p == self.door and not self.door_open: return 'door'
        if p == self.key and not self.has_key: return 'key'
        if p == self.switch: return 'switch'
        if p == self.crate: return 'crate'
        if p == self.goal: return 'goal'
        return 'empty'

    def _observe(self) -> np.ndarray:
        grid = np.zeros((7,7,3), dtype=np.uint8)
        for i in range(7):
            for j in range(7):
                p = (self.agent[0] + i-3, self.agent[1] + j-3) if self.partial else (i+1,j+1)
                grid[i,j] = COLORS[self._tile(p) if 0 <= p[0] < self.size and 0 <= p[1] < self.size else 'unknown']
        if self.noise > 0 or self.task == 'noise':
            rate = max(self.noise, 0.5 if self.task == 'noise' else 0)
            mask = self._rng.random((7,7)) < rate
            grid[mask] = self._rng.integers(0,256, size=(int(mask.sum()),3), dtype=np.uint8)
        return np.transpose(grid, (2,0,1)).copy()

    def step(self, action: int):
        if not self.action_space.contains(action): raise ValueError('Invalid action')
        self.steps += 1
        if action == 4:
            neighbours = {(self.agent[0] + dr,self.agent[1]+dc) for dr,dc in MOVES} | {self.agent}
            if self.switch in neighbours:
                self.switch_on = not self.switch_on
                if self.task == 'switch' or (self.task == 'hidden' and self._rule == 1):
                    self.door_open = self.switch_on
            if self.door in neighbours:
                if (self.task in ('key','altered','noise') and self.has_key or
                    self.task == 'switch' and self.switch_on or
                    self.task == 'hidden' and ((self._rule==0 and self.has_key) or (self._rule==1 and self.switch_on))):
                    self.door_open=True
        else:
            dr,dc = MOVES[int(action)]
            if (self.task == 'altered') ^ (self.reverse_after_key and self.has_key):
                dr,dc = -dr,-dc
            nxt = (self.agent[0]+dr,self.agent[1]+dc)
            if nxt == self.crate:
                pushed = (nxt[0]+dr,nxt[1]+dc)
                if pushed not in self.walls and pushed != self.door and 0 < pushed[0] < self.size-1 and 0 < pushed[1] < self.size-1:
                    self.crate = pushed
                else: nxt = self.agent
            if nxt not in self.walls and (nxt != self.door or self.door_open): self.agent = nxt
        if self.agent == self.key: self.has_key = True
        if self.task in ('key','altered','noise') and self.has_key and self.agent == self.door: self.door_open = True
        if self.task == 'hidden' and self._rule == 0 and self.has_key and self.agent == self.door: self.door_open = True
        done = self.agent == self.goal
        truncated = self.steps >= self.max_steps and not done
        observation = self._observe()
        self._last_observation = observation.copy()
        return observation, float(1.0 if done else -0.01), done, truncated, {'success': done}

    def render(self):
        if self.render_mode is None:
            return None
        # Rendering must not advance observation noise RNG or change the trajectory.
        return np.transpose(self._last_observation, (1,2,0)).copy()

    def close(self):
        return None

    def diagnostic_state(self):
        if not self._diagnostics: raise PermissionError('Diagnostic state is available only to evaluator-created environments')
        return {'rule': self._rule, 'agent': self.agent, 'has_key': self.has_key,
                'switch_on': self.switch_on, 'door_open': self.door_open}


def create_environment(task='key', **kwargs): return ProceduralWorld(task=task, **kwargs)
