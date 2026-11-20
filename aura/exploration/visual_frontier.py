"""Visual-odometry frontier explorer for bounded procedural environments.

The controller receives only image observations and its own issued actions. It
learns how movement actions displace its *relative* visual coordinate system by
matching translated images; all graph vertices are relative, online inferred
poses. It neither reads task diagnostics nor uses simulator tile/color tables.

Inductive priors: cardinal local camera translations, a declared set of
candidate navigation actions and a separate interaction action. These priors
are action-interface assumptions, not autonomous discovery of semantics.
Map inference can fail under identical/aliased views, camera rotation or heavy
noise. Its graph is reset per episode; it is not a general SLAM algorithm.
"""
from __future__ import annotations
from collections import Counter, deque
import numpy as np
from aura.exploration.validated_progress import ValidatedProgressExplorer

SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _match_fraction(previous: np.ndarray, current: np.ndarray, dy: int, dx: int) -> float:
    """Match current local view to previous shifted by (dy,dx).

    Missing boundary pixels are omitted rather than assigned wall colors.
    """
    a = np.asarray(previous, dtype=np.uint8)
    b = np.asarray(current, dtype=np.uint8)
    if a.shape != (3, 7, 7) or b.shape != (3, 7, 7):
        raise ValueError('Expected channel-first 7x7 RGB image')
    pa = a[:, max(dy, 0):7 + min(dy, 0), max(dx, 0):7 + min(dx, 0)]
    pb = b[:, max(-dy, 0):7 - max(dy, 0), max(-dx, 0):7 - max(dx, 0)]
    return float(np.mean(np.all(pa == pb, axis=0)))


def visual_displacement(previous: np.ndarray, current: np.ndarray,
                        margin: float = .06, min_match: float = .62):
    """Infer movement from unlabeled successive image observations.

    Returns zero when no shift is distinguishable with conservative margins;
    it may underestimate motion in repetitive or strongly noisy scenes.
    """
    scores = np.asarray([_match_fraction(previous, current, *shift) for shift in SHIFTS])
    k = int(np.argmax(scores))
    if k and scores[k] >= min_match and scores[k] - scores[0] >= margin:
        return SHIFTS[k]
    return (0, 0)


class VisualFrontierExplorer:
    """Build an observed action graph, navigate to its untried frontiers.

    `choose` is called once per step, followed by `observe` with the actual
    next observation. No reset/privileged state is accepted into `choose`.
    """
    def __init__(self, seed: int = 0, n_actions: int = 5,
                 navigation_actions=(0, 1, 2, 3), interaction_action=4,
                 epsilon: float = .04):
        navigation_actions = tuple(int(a) for a in navigation_actions)
        if (n_actions < 5 or len(navigation_actions) < 2
                or len(set(navigation_actions)) != len(navigation_actions)
                or not all(0 <= a < n_actions for a in navigation_actions)
                or interaction_action in navigation_actions
                or not 0 <= interaction_action < n_actions
                or not 0 <= epsilon <= 1):
            raise ValueError('Invalid action contract or exploration probability')
        self.rng = np.random.default_rng(seed)
        self.navigation_actions = navigation_actions
        self.interaction_action = interaction_action
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.predictor = ValidatedProgressExplorer(seed + 2999, n_actions=n_actions)
        self.action_shifts = {}  # discovered motion action -> observed camera shift
        self.shift_votes = {a: Counter() for a in navigation_actions}
        self.noise_probes = Counter()
        self.action_counts = np.zeros(n_actions, dtype=np.int64)
        self.experiences = 0
        self.motion_matches = 0
        self.episodes = 0
        self.reset_episode()

    def reset_episode(self):
        self.pose = (0, 0)
        self.visits = Counter({self.pose: 1})
        self.attempts = Counter()  # (relative pose, action)
        self.blocked = set()
        self.edges = {}  # observed directed (pose, action) -> pose
        self.interacted = set()  # pose/context signatures already probed
        self.last_interaction_pose = None
        self.last_interaction_step = -10000
        self.last_observation = None
        self.last_action = None
        self.episodes += 1
        self.episode_interactions = 0

    @staticmethod
    def _local_signature(observation):
        image = np.asarray(observation, dtype=np.uint8)
        # A local 3x3 visual neighborhood is the only evidence used to decide
        # whether a previously encountered place has changed after acting.
        return bytes(image[:, 2:5, 2:5].tobytes())

    @staticmethod
    def _local_distinctiveness(observation):
        image = np.asarray(observation, dtype=np.uint8)
        if image.shape != (3, 7, 7):
            raise ValueError('Expected channel-first 7x7 RGB image')
        palette = image.transpose(1, 2, 0).reshape(-1, 3)
        colors, inverse, counts = np.unique(palette, axis=0,
                                            return_inverse=True, return_counts=True)
        indices = inverse.reshape(7, 7)
        # No color labels: identify objects only through locally rare colors.
        neighbors = [indices[dy, dx] for dy, dx in ((2, 3), (3, 2), (3, 4), (4, 3))]
        return any(counts[n] <= 2 for n in neighbors)

    def _has_frontier(self, pose):
        return any(self.attempts[(pose, a)] == 0 and (pose, a) not in self.blocked
                   for a in self.navigation_actions)

    def _route_to_frontier(self):
        # Path search over the *observed* directed transition graph only.
        outgoing = {}
        for (pose, action), dest in self.edges.items():
            outgoing.setdefault(pose, []).append((action, dest))
        queue = deque([(self.pose, ())])
        seen = {self.pose}
        while queue:
            pose, route = queue.popleft()
            if pose != self.pose and self._has_frontier(pose):
                return route
            for action, dest in outgoing.get(pose, []):
                if dest not in seen:
                    seen.add(dest)
                    queue.append((dest, route + (action,)))
        return ()

    def choose(self, observation, *, policy='visual_frontier', epsilon=None, rng=None):
        if policy != 'visual_frontier':
            raise ValueError('Unsupported policy')
        chooser = self.rng if rng is None else rng
        explore = self.epsilon if epsilon is None else epsilon
        if not 0 <= explore <= 1:
            raise ValueError('Exploration probability must be in [0,1]')
        obs = np.asarray(observation, dtype=np.uint8)
        if obs.shape != (3, 7, 7):
            raise ValueError('Expected channel-first 7x7 RGB image')
        signature = (self.pose, self._local_signature(obs))
        # Interaction is retained at visually interesting states, including
        # states newly changed by acquiring an object or operating a switch.
        if (signature not in self.interacted and
                self._local_distinctiveness(obs) and
                self.attempts[(self.pose, self.interaction_action)] < 2 and
                (self.pose != self.last_interaction_pose or
                 self.episode_interactions - self.last_interaction_step >= 5)):
            action = self.interaction_action
            self.interacted.add(signature)
            self.last_interaction_pose = self.pose
            self.last_interaction_step = self.episode_interactions
        elif (self.episode_interactions >= 3 and
              any(self.action_counts[a] < 3 for a in range(self.n_actions)
                  if a not in self.navigation_actions and a != self.interaction_action)):
            # Probe other declared actions before concluding they are useless.
            # This is behavior-based testing, not hardcoded exclusion of TV.
            action = next(a for a in range(self.n_actions) if
                          a not in self.navigation_actions and
                          a != self.interaction_action and self.action_counts[a] < 3)
        elif chooser.random() < explore:
            action = int(chooser.choice(self.navigation_actions))
        else:
            # Prefer unseen one-step frontier cells. Learned action directions
            # are *not* supplied by the gridworld's MOVES table.
            candidates = []
            for a in self.navigation_actions:
                attempts = self.attempts[(self.pose, a)]
                if (self.pose, a) in self.blocked or attempts:
                    continue
                direction = self.action_shifts.get(a)
                target = ((self.pose[0] + direction[0], self.pose[1] + direction[1])
                          if direction else None)
                # Unknown actions are worth trying; known actions are most
                # valuable if they reach unvisited relative coordinates.
                novelty = 1.0 if target is None else 1.5 / (1 + self.visits[target])
                candidates.append((novelty, -self.action_counts[a], a))
            if candidates:
                best = max(candidates)[:2]
                tied = [a for score, count, a in candidates if (score, count) == best]
                action = int(chooser.choice(tied))
            else:
                route = self._route_to_frontier()
                if route:
                    action = int(route[0])
                else:
                    # Never get stuck: use known reversible edges when present
                    # and force fresh experiments after the graph is exhausted.
                    directions = list(self.navigation_actions)
                    action = int(min(directions, key=lambda a:
                                     (self.attempts[(self.pose, a)], self.action_counts[a])))
        self.track_action(obs, action)
        return action

    def track_action(self, observation, action):
        """Register an action selected externally by actor/skill/meta-control."""
        obs = np.asarray(observation, dtype=np.uint8)
        if obs.shape != (3, 7, 7) or not 0 <= int(action) < self.n_actions:
            raise ValueError('Invalid observed action/observation')
        if self.last_action is not None:
            raise ValueError('A previous action has not yet been observed')
        self.last_observation = obs.copy()
        self.last_action = int(action)
        self.action_counts[action] += 1
        self.episode_interactions += 1
        self.attempts[(self.pose, action)] += 1

    def observe(self, observation, action, next_observation):
        if self.last_action != action or self.last_observation is None:
            raise ValueError('Expected choose/observe action pair')
        previous = np.asarray(observation, dtype=np.uint8)
        if not np.array_equal(previous, self.last_observation):
            raise ValueError('The observed transition must match chosen observation')
        following = np.asarray(next_observation, dtype=np.uint8)
        if following.shape != (3, 7, 7):
            raise ValueError('Expected channel-first 7x7 RGB image')
        old = self.pose
        displacement = visual_displacement(previous, following)
        if action in self.navigation_actions:
            if displacement != (0, 0):
                self.shift_votes[action][displacement] += 1
                self.action_shifts[action] = displacement  # track the newest confirmed rule
                self.motion_matches += 1
                dest = (old[0] + displacement[0], old[1] + displacement[1])
                self.edges[(old, action)] = dest
                self.pose = dest
                self.visits[dest] += 1
                self.blocked.discard((old, action))
            else:
                self.blocked.add((old, action))
        elif displacement == (0, 0):
            self.noise_probes[action] += 1
            if action == self.interaction_action:
                # A changed mechanism can make previously blocked transitions
                # possible; invalidate failures only on stable scene changes.
                if self.predictor._stable_effect(previous, following) > .001:
                    reopened={edge for edge in self.blocked if edge[0]==old}
                    self.blocked.difference_update(reopened)
                    for edge in reopened:
                        self.attempts.pop(edge,None)
        self.predictor.observe(previous, action, following)
        self.last_action = None
        self.last_observation = None
        self.experiences += 1

    def features(self, image):
        return self.predictor.features(image)

    def prediction(self, action, features):
        return self.predictor.prediction(action, features)

    def metrics(self):
        return {
            'episode_count': self.episodes,
            'total_interactions': self.experiences,
            'action_counts': self.action_counts.tolist(),
            'inferred_action_displacements': {str(a): list(delta) for a, delta in self.action_shifts.items()},
            'observed_move_matches': self.motion_matches,
            'current_relative_poses': len(self.visits),
            'observed_directed_edges': len(self.edges),
            'nonmoving_probe_counts': dict(self.noise_probes),
            'representation': 'unlabeled image-shift visual odometry; no simulator state',
        }
