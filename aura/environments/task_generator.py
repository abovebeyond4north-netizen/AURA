"""Disjoint deterministic seed namespaces for experiments (not a security boundary)."""
import hashlib
from .procedural_world import TASKS

def task_seed(split: str, idx: int, secret: str = '') -> int:
    if split not in ('train', 'dev', 'protected'): raise ValueError('Invalid split')
    payload = f'{split}:{secret}:{idx}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')

def task_family(idx: int) -> str: return TASKS[idx % len(TASKS)]


def layout_spec(split, index):
    """Generate public, geometry-disjoint cases. This is not a secret test set."""
    columns = {'train': (2,4,6), 'dev': (3,5), 'protected': (1,7)}
    seed = task_seed(split,index)
    choices = columns[split]
    return {
        'seed': seed,
        'task': TASKS[index % len(TASKS)],
        'size': (9,11)[(index // len(TASKS)) % 2],
        'door_column': choices[(index // (len(TASKS)*2)) % len(choices)],
        'reverse_after_key': index % 2 == 1,
    }
