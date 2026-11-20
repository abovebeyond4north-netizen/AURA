"""Difficulty selection from recent competence, slope, and exploration diversity."""
from collections import deque
import numpy as np
from aura.environments.procedural_world import TASKS

class Curriculum:
    def __init__(self,tasks=TASKS,seed=0,window=20):
        self.tasks=tuple(tasks); self.history={t:deque(maxlen=window) for t in tasks}
        self.rng=np.random.default_rng(seed); self.choices=[]
    def update(self,task,success): self.history[task].append(float(bool(success)))
    def choose_goal(self):
        scores=[]
        for task in self.tasks:
            h=np.array(self.history[task]); competence=h.mean() if len(h) else .5
            slope=abs(h[len(h)//2:].mean()-h[:len(h)//2].mean()) if len(h)>=4 else .25
            # penalize extremes and reward improvement near frontier
            score=.65*(1-abs(competence-.5)*2)+.25*slope+.1/np.sqrt(1+len(h))
            scores.append(max(score,0.02))
        p=np.asarray(scores); p=p/p.sum()
        unattempted=[i for i,t in enumerate(self.tasks) if not self.history[t]]
        if unattempted:
            chosen=self.tasks[int(self.rng.choice(unattempted))]
            reason='unattempted task coverage before competence-weighted selection'
        else:
            chosen=self.tasks[int(self.rng.choice(len(self.tasks),p=p))]
            reason='competence progress, frontier and diversity'
        self.choices.append({'task':chosen,'scores':dict(zip(self.tasks,map(float,scores))),
                             'reason':reason})
        return chosen


class PrerequisiteCurriculum(Curriculum):
    """Budget practice using demonstrated readiness and recent success."""

    def __init__(self, prerequisites, seed=0):
        super().__init__(tasks=prerequisites, seed=seed)
        self.prerequisites = prerequisites
        self.attempts = dict.fromkeys(self.tasks, 0)

    def mastered(self, task):
        history = self.history[task]
        return len(history) >= 3 and all(list(history)[-3:])

    def update(self, task, success):
        super().update(task, success)
        self.attempts[task] += 1

    def choose_goal(self):
        ready = [t for t in self.tasks if all(self.mastered(p) for p in self.prerequisites[t])]
        if not ready:
            raise ValueError('No curriculum task has satisfied prerequisites')
        pending = [t for t in ready if not self.mastered(t)]
        candidates = pending or ready
        scores = {}
        factors = {}
        for task in candidates:
            history = list(self.history[task])
            difficulty = 1 - (sum(history) + 1)/(len(history) + 2)
            progress = float(np.mean(history[-3:])-np.mean(history[:-3])) if len(history)>3 else 0.
            transfer = sum(task in parents and not self.mastered(child)
                           for child, parents in self.prerequisites.items())
            novelty = 1 / (1+self.attempts[task])
            failed = bool(history and not history[-1])
            score = novelty + max(0., progress) + .5*transfer + .2*failed - .25*difficulty
            factors[task] = dict(novelty=novelty, progress=progress, transfer=transfer,
                                 last_failed=failed, predicted_difficulty=difficulty)
            scores[task] = score
        tied = [t for t in candidates if scores[t] == max(scores.values())]
        chosen = str(self.rng.choice(tied))
        self.choices.append(dict(task=chosen, readiness=ready, scores=scores, factors=factors,
                                 reason='readiness, mastery, novelty, recent progress and transfer'))
        return chosen
