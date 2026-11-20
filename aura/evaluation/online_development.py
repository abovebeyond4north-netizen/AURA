"""Fixed local development data, separate from replay and protected evaluation."""
from __future__ import annotations
import hashlib
import numpy as np
from aura.environments.procedural_world import create_environment
from aura.environments.task_generator import task_seed

DEVELOPMENT_TRANSITIONS = 8


def collect_online_development(task: str, seed_index: int):
    """One fixed random-policy rollout; no model, replay or protected seeds."""
    seed = task_seed('dev', seed_index)
    episode_seeds = [seed]
    rng = np.random.default_rng(seed + 9)
    env = create_environment(task, max_steps=DEVELOPMENT_TRANSITIONS,
                             noise=.5 if task == 'noise' else .05)
    x, a, y = [], [], []
    try:
        obs, _ = env.reset(seed=seed)
        for i in range(DEVELOPMENT_TRANSITIONS):
            action = int(rng.integers(5))
            nxt, _, done, truncated, _ = env.step(action)
            x.append(obs.copy()); a.append(action); y.append(nxt.copy())
            obs = nxt
            if (done or truncated) and i + 1 < DEVELOPMENT_TRANSITIONS:
                next_seed = task_seed('dev', seed_index * DEVELOPMENT_TRANSITIONS + i + 1)
                episode_seeds.append(next_seed)
                obs, _ = env.reset(seed=next_seed)
    finally:
        close = getattr(env, 'close', None)
        if close is not None:
            close()
    batch = (np.stack(x), np.asarray(a, dtype=np.int64), np.stack(y))
    digest = hashlib.sha256(b''.join(arr.tobytes() for arr in batch)).hexdigest()
    metadata = {'seed': seed, 'episode_seeds': episode_seeds, 'task': task, 'transitions': len(a),
                'sha256': digest, 'scope': 'fixed local development; not independent promotion evidence'}
    return batch, metadata
