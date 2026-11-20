"""Learn an egocentric two-action doorway option from observed trials."""
from collections import defaultdict
from dataclasses import dataclass

from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import offset
from aura.exploration.visual_frontier import visual_displacement


@dataclass(frozen=True)
class DoorOption:
    actions: tuple
    evidence_ids: tuple
    successes: int
    failures: int

    def execute(self, camera, direction):
        """Abort if the initiation cue or an expected intermediate state fails."""
        delta = MOVES[direction]
        if camera.at_pixel(delta) != 'door':
            return False
        before = camera.image
        middle = camera.step(resolve(self.actions[0], direction))
        camera.image = middle
        if visual_displacement(before, middle) != (0, 0) or camera.at_pixel(delta) != 'empty':
            return False
        after = camera.step(resolve(self.actions[1], direction))
        if visual_displacement(middle, after) != delta:
            return False
        camera.position = offset(camera.position, delta)
        camera._observe(after)
        return True


def resolve(token, direction):
    # Forward follows the observed doorway direction.
    return direction if token == 'forward' else token


def learn(samples, minimum=3):
    """Accept a two-action sequence supported by distinct successful episodes."""
    groups = defaultdict(dict)
    seen = set()
    for sequence, success, evidence_id in samples:
        if evidence_id in seen:
            raise ValueError('Duplicate observed episode')
        seen.add(evidence_id)
        groups[tuple(sequence)][evidence_id] = success
    candidates = []
    for actions, rows in groups.items():
        wins = sum(rows.values())
        if wins >= minimum and wins == len(rows):
            candidates.append(DoorOption(actions, tuple(rows), wins, 0))
    return max(candidates, key=lambda option: option.successes, default=None)


class OptionReliability:
    """Track context-level outcomes and withdraw after repeated failures."""
    def __init__(self):
        self.outcomes = defaultdict(dict)

    def available(self, context):
        return list(self.outcomes[context].values())[-3:] != [False]*3

    def record(self, context, evidence_id, succeeded):
        outcomes = self.outcomes[context]
        if evidence_id in outcomes:
            raise ValueError('Duplicate evaluation episode')
        outcomes[evidence_id] = bool(succeeded)
