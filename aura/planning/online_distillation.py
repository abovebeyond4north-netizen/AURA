"""Bounded, opt-in shadow neural training; never activates a candidate."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import torch
from aura.environments.procedural_world import create_environment
from aura.environments.task_generator import task_seed
from aura.planning.distillation import RecurrentImitation, rewarded_episodes, split_episodes, fit_imitation
from aura.types import Source


def ancestral_rehearsal(episodes, lineage, limit):
    """Resolve only prior TRAIN episode IDs, never ancestral validation groups."""
    if type(limit) is not int or not 0 <= limit <= 16:
        raise ValueError('Invalid rehearsal limit')
    if not limit:
        return []
    trained = {tuple(ids) for r in lineage for ids in r.get('presented_training_evidence', r['training_evidence'])}
    held = {seed for r in lineage for seed in r['validation_seeds']}
    eligible = [ep for ep in episodes if ep[0].seed not in held
                and tuple(e.evidence_id for e in ep) in trained and len(ep) <= 256]
    # Prefer recent distinct seed groups, bounding total training tensors.
    selected = []
    seen = set()
    for ep in reversed(eligible):
        if ep[0].seed not in seen:
            selected.append(ep)
            seen.add(ep[0].seed)
            if len(selected) == limit:
                break
    return list(reversed(selected))


class OnlineDistillation:
    def __init__(self, config, ledger, budget, output):
        self.config, self.ledger, self.budget = config, ledger, budget
        self.output = Path(output)
        self.reports = []
        self.pending = {}
        self.last_episode = None
        self.evaluation_seeds = set()
        self.optimizer_updates = 0

    def observe(self, final, *, step):
        """Called once after a committed terminal transition, never mid-episode."""
        c = self.config
        if not c.imitation_interval or not final.terminated or final.truncated or final.reward <= 0:
            return
        if self.last_episode == final.episode_id:
            return
        self.last_episode = final.episode_id
        task = final.context
        self.pending[task] = self.pending.get(task, 0) + 1
        if self.pending[task] < c.imitation_interval or len(self.reports) >= c.imitation_max_candidates:
            return
        horizon = c.max_episode_steps
        cap = 2 * c.imitation_eval_episodes * horizon
        if not (self.budget.can_spend('candidates') and self.budget.can_spend('updates', c.imitation_updates)
                and self.budget.can_spend('imitation_eval_interactions', cap)):
            return
        # Ledger is committed by the same single-threaded agent before this call.
        # Limit tensor allocation even when the run's total ledger is large.
        episodes = [ep for ep in rewarded_episodes(self.ledger) if ep[0].context == task
                    and len(ep) <= 256]
        all_episodes = episodes
        lineage = []
        parent = None
        parent_bytes = None
        if c.imitation_inherit:
            lineage = [r for r in self.reports if r['task'] == task]
            if lineage:
                parent = lineage[-1]
                # Require the entire episode to have been acquired after its parent,
                # and exclude all previously used seed groups in this task lineage.
                marker = f"imitation-candidate:{parent['candidate']}"
                records = self.ledger.records()
                matches = [i for i, r in enumerate(records) if r['record_id'] == marker]
                if (len(matches) != 1 or records[matches[0]]['kind'] != 'imitation_candidate'
                        or records[matches[0]]['payload'] != parent):
                    raise ValueError('Parent evidence mismatch')
                new_ids = {r['record_id'] for r in records[matches[0]+1:] if r['kind'] == 'experience'}
                old_groups = {seed for r in lineage for seed in r['training_seeds'] + r['validation_seeds']}
                episodes = [ep for ep in episodes if ep[0].seed not in old_groups
                            and all(e.evidence_id in new_ids for e in ep)]
        episodes = episodes[-32:]
        if len({ep[0].seed for ep in episodes}) < 2:
            return
        index = self.budget.used['candidates'] + 1
        seed = (c.seed + index) % (2**63 - 1)
        train, val = split_episodes(episodes, seed)
        fresh_train = train
        rehearsal = ancestral_rehearsal(all_episodes, lineage,
                                        min(c.imitation_rehearsal_episodes, 32-len(train)))
        train = fresh_train + rehearsal
        seen = {r['payload']['seed'] for r in self.ledger.records('experience')} | self.evaluation_seeds
        seeds = [task_seed('dev', 4_000_000_000 + index * 100 + i, secret=f'imitation:{c.seed}')
                 for i in range(c.imitation_eval_episodes)]
        if len(set(seeds)) != len(seeds) or seen.intersection(seeds):
            raise RuntimeError('Imitation evaluation seed collision')
        if parent is not None:
            expected_name = f"imitation-candidate-{parent['candidate']:04d}.pt"
            if parent['checkpoint'] != expected_name:
                raise ValueError('Parent checkpoint name mismatch')
            parent_bytes = (self.output / expected_name).read_bytes()
            if hashlib.sha256(parent_bytes).hexdigest() != parent['checkpoint_sha256']:
                raise ValueError('Parent checkpoint digest mismatch')
        self.evaluation_seeds.update(seeds)
        self.budget.spend('candidates')
        self.budget.spend('updates', c.imitation_updates)
        self.optimizer_updates += c.imitation_updates
        # CPU-only shadow controller must not alter the training agent's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = (RecurrentImitation.load(io.BytesIO(parent_bytes))
                     if parent_bytes is not None else RecurrentImitation())
            before = self._evaluate(model, task, seeds)
            fit = fit_imitation(model, train, val, updates=c.imitation_updates,
                                budgeted_fresh_count=len(fresh_train) if c.imitation_budgeted_rehearsal else None)
            after = self._evaluate(model, task, seeds)
        name = f'imitation-candidate-{index:04d}.pt'
        path = self.output / name
        if path.exists():
            raise FileExistsError(path)
        model.save(path)
        report = {'candidate': index, 'step': step, 'task': task, 'checkpoint': name,
                  'parent_candidate': parent['candidate'] if parent else None,
                  'parent_checkpoint_sha256': parent['checkpoint_sha256'] if parent else None,
                  'generation': parent['generation'] + 1 if parent else 1,
                  'baseline': 'parent_checkpoint' if parent else 'random_initialization',
                  'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'training_seeds': sorted({ep[0].seed for ep in train}),
                  'validation_seeds': sorted({ep[0].seed for ep in val}),
                  'fresh_training_seeds': sorted({ep[0].seed for ep in fresh_train}),
                  'fresh_training_evidence': [[e.evidence_id for e in ep] for ep in fresh_train],
                  'rehearsal_seeds': sorted({ep[0].seed for ep in rehearsal}),
                  'rehearsal_evidence': [[e.evidence_id for e in ep] for ep in rehearsal],
                  'training_evidence': [[e.evidence_id for e in ep] for ep in train],
                  'validation_evidence': [[e.evidence_id for e in ep] for ep in val],
                  'evaluation_seeds': seeds, 'training': fit, 'before': before, 'after': after,
                  'evaluation_interactions': sum(r['steps'] for r in before + after),
                  'training_transition_presentations': fit['training_transition_presentations'],
                  'training_padded_step_presentations': fit['training_padded_step_presentations'],
                  'rehearsal_transition_presentations': sum(len(ep)*n for ep,n in zip(rehearsal, fit['episode_presentations'][len(fresh_train):])),
                  'measurement_padded_step_presentations': fit['measurement_padded_step_presentations'],
                  'presented_training_evidence': [[e.evidence_id for e in ep] for ep,n in zip(train,fit['episode_presentations']) if n],
                  'promotion': 'not_evaluated_by_protected_verifier',
                  'scope': ('public development shadow evaluation; no activation; fresh task-local seed groups after parent; only prior training episodes eligible for bounded rehearsal'
                            if c.imitation_inherit else
                            'public development shadow evaluation; no activation; repeated training data across candidates allowed')}
        self.ledger.append(f'imitation-candidate:{index}', 'imitation_candidate', Source.INFERRED, report)
        self.reports.append(report)
        self.pending[task] = 0

    def _evaluate(self, model, task, seeds):
        rows = []
        for seed in seeds:
            env = create_environment(task=task, max_steps=self.config.max_episode_steps,
                                     noise=.5 if task == 'noise' else .05)
            try:
                obs, _ = env.reset(seed=seed)
                model.reset_episode()
                total = 0.
                for step in range(1, self.config.max_episode_steps + 1):
                    self.budget.spend('imitation_eval_interactions')
                    obs, reward, done, trunc, _ = env.step(model.choose(obs))
                    total += reward
                    if done or trunc:
                        break
                rows.append({'seed': seed, 'steps': step, 'success': bool(done), 'return': total})
            finally:
                close = getattr(env, 'close', None)
                if close is not None:
                    close()
        return rows
