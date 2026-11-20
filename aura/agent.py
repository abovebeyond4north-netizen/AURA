"""End-to-end bounded grounded learner; the evaluation kernel is separate."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
from pathlib import Path
import json, random, time, uuid, sys
import numpy as np
import torch
from aura.config import Config, seed_everything
from aura.types import Experience, Source
from aura.world_model.dynamics import WorldModel
from aura.world_model.temporal_heads import TemporalHeads
from aura.world_model.uncertainty import UncertaintyEstimator
from aura.exploration.curiosity import Curiosity, LearningProgress
from aura.exploration.validated_progress import ValidatedProgressExplorer
from aura.exploration.visual_frontier import VisualFrontierExplorer
from aura.exploration.reward_routes import RewardRouteExplorer
from aura.cognition.hypotheses import HypothesisEngine
from aura.cognition.concepts import ConceptDiscovery
from aura.skills.discovery import SkillDiscovery
from aura.curriculum.generator import Curriculum
from aura.meta_learning.controller import MetaController
from aura.meta_learning.resource_allocator import Budget
from aura.memory.replay import ExperienceReplay
from aura.memory.evidence_ledger import EvidenceLedger
from aura.memory.episodic import EpisodicMemory
from aura.memory.semantic import SemanticMemory
from aura.memory.procedural import ProceduralMemory
from aura.planning.actor import Actor
from aura.planning.online_distillation import OnlineDistillation
from aura.planning.planner import Planner
from aura.environments.procedural_world import create_environment, TASKS
from aura.environments.task_generator import task_seed
from aura.evaluation.online_development import collect_online_development, DEVELOPMENT_TRANSITIONS

class AuraOmega:
    def __init__(self,config:Config,output_dir=None):
        seed_everything(config.seed)
        self.config=config
        self.rng=np.random.default_rng(config.seed)
        self.world_model=WorldModel(config.latent_dim,config.hidden_dim,config.ensemble_size,config.lr,config.torch_device)
        self.temporal_heads=TemporalHeads(self.world_model.encoder,config.latent_dim,(1,2,4),
                                          hidden_dim=config.hidden_dim,lr=config.lr,
                                          device=config.torch_device,freeze_encoder=True,mode='residual')
        self.temporal_updates=0
        self.actor=Actor(config.latent_dim,config.lr,config.torch_device)
        self.replay=ExperienceReplay()
        self.curiosity=Curiosity(self.world_model)
        self.progress_explorer=(ValidatedProgressExplorer(seed=config.seed)
                                if config.curiosity_policy=='validated_progress' else None)
        self.visual_frontier=(VisualFrontierExplorer(seed=config.seed)
                              if config.curiosity_policy=='visual_frontier' else None)
        self.reward_routes=(RewardRouteExplorer(seed=config.seed)
                            if config.curiosity_policy=='reward_routes' else None)
        self.explorer=(self.reward_routes if self.reward_routes is not None else
                       self.visual_frontier if self.visual_frontier is not None else
                       self.progress_explorer if self.progress_explorer is not None else self.curiosity)
        self.uncertainty=UncertaintyEstimator(self.world_model)
        self.hypotheses=HypothesisEngine()
        self.concepts=ConceptDiscovery()
        self.skills=SkillDiscovery()
        self.curriculum=Curriculum(seed=config.seed)
        self.meta=MetaController(('EXPLORE','PLAN','PROBE','ACTOR','PRACTICE_SKILL'))
        self.planner=Planner(self.world_model,self.actor,self.explorer,uncertainty_estimator=self.uncertainty)
        self.output_dir=Path(output_dir or config.log_dir)
        if (self.output_dir/'checkpoint.pt').exists():
            self.output_dir=self.output_dir/f'run-{uuid.uuid4().hex[:8]}'
        self.output_dir.mkdir(parents=True,exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard=SummaryWriter(log_dir=str(self.output_dir/'tensorboard'))
        except ImportError:
            # Offline CPU test image lacks TensorBoard; structured JSON remains authoritative.
            self.tensorboard=None
        self.ledger=EvidenceLedger(self.output_dir/'evidence.sqlite')
        self.episodic=EpisodicMemory(self.ledger)
        self.semantic=SemanticMemory(self.ledger)
        self.procedural=ProceduralMemory(self.ledger)
        self.budget=Budget(interactions=config.steps,updates=3*config.steps,seconds=3600,
                           development_interactions=DEVELOPMENT_TRANSITIONS*len(TASKS),
                           candidates=config.imitation_max_candidates if config.imitation_interval else 0,
                           imitation_eval_interactions=(2*config.imitation_max_candidates*config.imitation_eval_episodes*config.max_episode_steps
                                                        if config.imitation_interval else 0))
        self.imitation=OnlineDistillation(config,self.ledger,self.budget,self.output_dir)
        self._development_batch=None; self._development_metadata=None
        self._development_banks={}; self._progress_by_task={}; self._active_development_task=None
        self.history=[]; self.concept_trials=0; self.accepted_concepts=0
        # Live continuation (not a process-restart checkpoint).
        self._env=None; self._obs=None; self._episode=[]; self._episode_num=0
        self._episode_wins=[]; self._model_losses=[]; self._episode_task=None
        self._episode_seed=None; self._episode_condition=None; self._requested_task=None
        self._skill_queue=deque(); self._active_skill=None
        self._wall_seconds=0.; self._cpu_seconds=0.

    @classmethod
    def initialize(cls,pretrained=False,seed=42,config=None,output_dir=None):
        if pretrained: raise ValueError('Only random initialization is supported')
        conf=(config or Config()).model_copy(update={'seed':seed})
        return cls(conf,output_dir)

    def train(self,steps=None,task=None,log_every=25,condition="omega"):
        if condition not in ("random", "fixed", "curiosity", "adaptive", "omega"):
            raise ValueError('Unknown experiment condition')
        if self.reward_routes is not None and condition!='curiosity':
            raise ValueError('Experimental reward_routes policy requires curiosity condition')
        if steps is not None and (type(steps) is not int or not 1<=steps<=self.config.steps):
            raise ValueError('steps must be a cumulative target within the configured budget')
        if type(log_every) is not int or log_every<1: raise ValueError('Invalid logging interval')
        steps=self.config.steps if steps is None else steps
        if task is not None and task not in TASKS: raise ValueError('Unknown task')
        if self._env is not None and (condition!=self._episode_condition or task!=self._requested_task):
            raise ValueError('Change condition/task only at a completed episode boundary')
        env=self._env; obs=self._obs; episode=self._episode; episode_num=self._episode_num
        episode_wins=self._episode_wins; model_losses=self._model_losses
        start=time.monotonic(); cpu_start=time.process_time()
        task_name=self._episode_task; seed=self._episode_seed
        skill_queue=self._skill_queue; active_skill=self._active_skill
        while len(self.history)<steps and self.budget.remaining():
            if env is None:
                task_name=task or (self.curriculum.choose_goal() if condition in ("adaptive","omega") else "key")
                seed=task_seed('train',episode_num)
                if seed in self.imitation.evaluation_seeds:
                    raise RuntimeError('Training seed overlaps shadow evaluation')
                env=create_environment(task=task_name,max_steps=self.config.max_episode_steps,
                                       noise=.05 if task_name!='noise' else .5)
                obs,_=env.reset(seed=seed)
                if self.visual_frontier is not None:
                    self.visual_frontier.reset_episode()
                if self.reward_routes is not None:
                    self.reward_routes.reset_episode()
                episode=[]; skill_queue.clear(); active_skill=None
            self._select_development_task(task_name)
            self.budget.spend('interactions')
            available=['EXPLORE','PLAN','PROBE','ACTOR']
            usable=[skill for skill in self.skills.skill_library.values() if skill.applicable(task_name)]
            if condition in ('adaptive','omega') and usable: available.append('PRACTICE_SKILL')
            # Concept refinement is periodic, not an action-selection mode.
            mode=None
            if condition in ('adaptive','omega') and not skill_queue and len(self.replay)>=self.config.warmup:
                mode=self.meta.select(allowed=available)
            if self.reward_routes is not None:
                action=self.reward_routes.choose(obs)
            elif skill_queue:
                action=skill_queue.popleft()
            elif mode=='PRACTICE_SKILL' and usable:
                active_skill=usable[int(self.rng.integers(len(usable)))]
                skill_queue.extend(active_skill.actions)
                action=skill_queue.popleft()
            elif condition=='random': action=int(self.rng.integers(5))
            elif condition=='fixed': action=self.actor.select(self.world_model,obs,self.config.epsilon,self.rng)
            elif mode=='PROBE' and condition in ('adaptive','omega'):
                action=4 if self.rng.random()<.6 else self.explorer.choose(obs,epsilon=self.config.epsilon,rng=self.rng)
            elif mode=='EXPLORE' or condition=='curiosity' or len(self.replay)<self.config.warmup:
                action=self.explorer.choose(obs,epsilon=self.config.epsilon,rng=self.rng)
            elif mode=='PLAN' and condition in ('adaptive','omega'):
                action,_=self.planner.choose(obs,self.rng)
            else: action=self.actor.select(self.world_model,obs,self.config.epsilon,self.rng)
            if self.visual_frontier is not None and self.visual_frontier.last_action is None:
                # Keep the map synchronized when actor/skill/meta-controller acts.
                self.visual_frontier.track_action(obs,action)
            next_obs,reward,done,trunc,info=env.step(action)
            if self.visual_frontier is not None:
                self.visual_frontier.observe(obs,action,next_obs)
            if self.progress_explorer is not None:
                self.progress_explorer.observe(obs,action,next_obs)
            exp=Experience(obs.copy(),action,next_obs.copy(),reward,done,trunc,
                           str(episode_num),str(uuid.uuid4()),seed=seed,context=task_name)
            self.replay.add(exp); self.episodic.add(exp); episode.append(exp)
            if self.reward_routes is not None:
                self.reward_routes.observe(obs,action,next_obs,reward=reward,terminated=done,
                                           truncated=trunc,evidence_id=exp.evidence_id)
            if len(self.replay)>=60 and not self.uncertainty.shift_is_calibrated:
                # Training-only observed images; the calibration half is
                # disjoint from the reference half and never uses test seeds.
                reference=list(self.replay.data)[:60]
                fit=self.uncertainty.fit_shift_detector([r.observation for r in reference])
                self.ledger.append('shift-fit:'+str(uuid.uuid4()),'calibration',Source.INFERRED,
                                   {'method':'frozen_empirical_nearest_reference',
                                    'calibration':fit,
                                    'evidence_ids':[r.evidence_id for r in reference]})
            # Intentionally no hidden diagnostic state used for training.
            if action==4:
                visible_change=bool(np.mean(np.abs(next_obs.astype(float)-obs.astype(float)))>.1)
                hyp=self.hypotheses.update(task_name,action,visible_change,exp.evidence_id,intervened=False)
                self.semantic.revise(hyp)
            fresh_gain=None
            if (condition != "random" and len(self.replay)>=max(self.config.warmup,self.config.batch_size)
                    and self.budget.can_spend('updates',2)):
                if (self._development_batch is None and
                        self.budget.can_spend('development_interactions',DEVELOPMENT_TRANSITIONS)):
                    # Reserve the bounded collection before sampling an independent environment.
                    self.budget.spend('development_interactions',DEVELOPMENT_TRANSITIONS)
                    self._development_batch,self._development_metadata=collect_online_development(task_name,self.config.seed)
                    self._development_banks[task_name]=(self._development_batch,self._development_metadata)
                    self.ledger.append('development:'+str(uuid.uuid4()),'development_validation',
                                       Source.OBSERVED,self._development_metadata)
                # Sampling by mode count avoids phase-locking with cyclic UCB selection.
                scheduled=(len(self.history)%10==0 if mode is None else
                           int(self.meta.counts[self.meta.strategies.index(mode)])%10==0)
                measure=(self._development_batch is not None and
                         (scheduled or not self.curiosity.progress.errors))
                before=self.world_model.prediction_loss(self._development_batch) if measure else None
                batch=self.replay.sample(self.config.batch_size)
                self.budget.spend('updates')
                result=self.world_model.train_batch(batch)
                model_losses.append(result['loss'])
                if len(self.replay)>=self.config.batch_size:
                    self.budget.spend('updates')
                    actor_loss=self.actor.train_batch(self.world_model,random.sample(list(self.replay.data),self.config.batch_size))
                if measure:
                    after=self.world_model.prediction_loss(self._development_batch)
                    fresh_gain=before-after
                    self.curiosity.update_validation(after)
                    self.ledger.append('gain:'+str(uuid.uuid4()),'development_gain',Source.INFERRED,{
                        'task':task_name,'bank_sha256':self._development_metadata['sha256'],
                        'step':len(self.history)+1,'before':before,'after':after,'gain':fresh_gain,
                        'selected_mode':mode,'scope':'paired local model-update delta; not causal policy credit'})
            if mode is not None:
                self.meta.update(mode,fresh_gain,1)
            self.history.append({'step':len(self.history)+1,'reward':reward,'task':task_name,
                                 'success':done,'uncertainty':float(self.world_model.estimate_uncertainty(obs[None],np.array([action]))[0])})
            obs=next_obs
            if done or trunc:
                if condition!='random' and self.config.temporal_updates_per_episode and len(episode)>=4:
                    temporal_trajectory=(np.stack([episode[0].observation]+[e.next_observation for e in episode]),
                                         [e.action for e in episode],
                                         [e.terminated or e.truncated for e in episode])
                    available_updates=max(0,self.budget.limits['updates']-self.budget.used['updates'])
                    n_updates=min(self.config.temporal_updates_per_episode,available_updates)
                    if n_updates:
                        result_temporal=self.temporal_heads.train_trajectories([temporal_trajectory],updates=n_updates)
                        self.budget.spend('updates',n_updates)
                        self.temporal_updates+=result_temporal['updates']
                episode_wins.append(bool(done)); self.curriculum.update(task_name,done)
                if condition in ('adaptive','omega') or self.reward_routes is not None:
                    for sid in self.skills.observe_episode(episode,done,task_name):
                        self.procedural.revise(self.skills.skill_library[sid].skill)
                if condition!='random':
                    self.imitation.observe(exp,step=len(self.history))
                episode_num+=1
                close=getattr(env,'close',None)
                if close is not None: close()
                env=None
            if active_skill is not None and (not skill_queue or done or trunc):
                if done: active_skill.skill.successes+=1
                else: active_skill.skill.failures+=1
                self.procedural.revise(active_skill.skill)
                if active_skill.skill.failures>10 and active_skill.skill.successes==0:
                    self.skills.skill_library.pop(active_skill.skill.skill_id,None)
                active_skill=None
                skill_queue.clear()
            if (condition in ('adaptive','omega') and len(self.replay)>=50 and
                    len(self.history)%50==0 and self.budget.can_spend('updates')):
                self._refine_concepts()
            if len(self.history)%5==0:
                self.uncertainty.update(exp.observation,exp.action,exp.next_observation)
            if len(self.history)%log_every==0:
                self._log({'step':len(self.history),'world_loss':model_losses[-1] if model_losses else None,
                           'episodes':episode_num,'success_rate':float(np.mean(episode_wins)) if episode_wins else None})
        self._env=env; self._obs=obs; self._episode=episode; self._episode_num=episode_num
        self._episode_task=task_name; self._episode_seed=seed
        self._episode_condition=condition; self._requested_task=task
        self._active_skill=active_skill
        self.world_model.save(self.output_dir/'checkpoint.pt',{'seed':self.config.seed,'config':self.config.model_dump()})
        self.temporal_heads.save(self.output_dir/'temporal.pt',update_steps=self.temporal_updates)
        self.actor.save(self.output_dir/'actor.pt')
        if self.progress_explorer is not None or self.visual_frontier is not None or self.reward_routes is not None:
            exploration_metrics=self.explorer.metrics()
            self.ledger.append('exploration:'+str(uuid.uuid4()),'exploration_summary',
                               Source.INFERRED,{'method':self.config.curiosity_policy,
                                                'metrics':exploration_metrics,
                                                'evidence_ids':[e.evidence_id for e in self.replay.data]})
        else:
            exploration_metrics=None
        self._wall_seconds+=time.monotonic()-start; self._cpu_seconds+=time.process_time()-cpu_start
        resource_metrics=self._resource_metrics(self._wall_seconds,self._cpu_seconds)
        self._log({'complete':True,'steps':len(self.history),'episodes':episode_num,
                   'success_rate':float(np.mean(episode_wins)) if episode_wins else None,
                   'last_training_loss':model_losses[-1] if model_losses else None,
                   'ledger_valid':self.ledger.verify(),'elapsed_seconds':time.monotonic()-start,
                   'discovered_skills':len(self.skills.skill_library),
                   'accepted_concepts':self.accepted_concepts,'concept_trials':self.concept_trials,
                   'temporal_training_steps':self.temporal_updates,
                   'uncertainty_calibration':self.uncertainty.calibration(),
                   'shift_detector_fitted':self.uncertainty.shift_is_calibrated,
                   'exploration_policy':self.config.curiosity_policy,
                   'exploration_metrics':exploration_metrics,
                   'resource_metrics':resource_metrics,
                   'imitation_candidates':self.imitation.reports,
                   'online_development_validation':self._development_metadata,
                   'development_validation_by_task':{t:m for t,(_,m) in self._development_banks.items()},
                   'meta_selection_metrics':self.meta.metrics(),
                   'mean_reward':float(np.mean([r['reward'] for r in self.history]))})
        return {'steps':len(self.history),'episodes':episode_num,
                'last_training_loss':model_losses[-1] if model_losses else None,
                'success_rate':float(np.mean(episode_wins)) if episode_wins else None,
                'discovered_skills':len(self.skills.skill_library),
                'accepted_concepts':self.accepted_concepts,
                'temporal_training_steps':self.temporal_updates,
                'resource_metrics':resource_metrics,
                   'imitation_candidates':self.imitation.reports,
                   'online_development_validation':self._development_metadata,
                   'development_validation_by_task':{t:m for t,(_,m) in self._development_banks.items()},
                   'meta_selection_metrics':self.meta.metrics(),
                'exploration_policy':self.config.curiosity_policy,
                'exploration_metrics':exploration_metrics,
                'condition':condition,
                'ledger_valid':self.ledger.verify(),
                'actor_checkpoint':str(self.output_dir/'actor.pt'),
                'checkpoint':str(self.output_dir/'checkpoint.pt')}
    def _select_development_task(self,task):
        """Restore task-local feedback; never compare errors across different banks."""
        if task==self._active_development_task: return
        if self._active_development_task is not None:
            self._progress_by_task[self._active_development_task]=(self.curiosity.progress,
                self.curiosity.validated_progress,self.curiosity.irreducible_error)
        (self.curiosity.progress,self.curiosity.validated_progress,self.curiosity.irreducible_error)=self._progress_by_task.get(task,(LearningProgress(),0.,0.))
        self._development_batch,self._development_metadata=self._development_banks.get(task,(None,None))
        self._active_development_task=task

    def _refine_concepts(self):
        # Predictive proxies from actual replay only; never semantic labels.
        experiences=random.sample(list(self.replay.data),min(100,len(self.replay)))
        with torch.no_grad():
            z=self.world_model.encoder(self.world_model._tensor(np.stack([e.observation for e in experiences]))).cpu().numpy()
            next_z=self.world_model.encoder(self.world_model._tensor(np.stack([e.next_observation for e in experiences]))).cpu().numpy()
        self.concept_trials+=1
        self.budget.spend('updates')
        outcome=self.concepts.fit(z,next_z)
        if outcome['accepted']:
            self.accepted_concepts+=1
            self.ledger.append('concept:'+str(uuid.uuid4()),'concept',Source.INFERRED,{
                **outcome,'method':'latent clustering predicting next latent',
                'evidence_ids':[e.evidence_id for e in experiences]})

    def _resource_metrics(self,wall,cpu):
        result={'wall_seconds':float(wall),'cpu_seconds':float(cpu),
                'environment_interactions':int(self.budget.used['interactions']),
                'development_interactions':int(self.budget.used['development_interactions']),
                'imitation_evaluation_interactions':int(self.budget.used['imitation_eval_interactions']),
                'imitation_optimizer_updates':self.imitation.optimizer_updates,
                'total_environment_interactions':int(self.budget.used['interactions']+self.budget.used['development_interactions']+self.budget.used['imitation_eval_interactions']),
                'optimization_or_concept_steps':int(self.budget.used['updates'])}
        try:
            import resource
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['peak_rss_mib']=float(rss/(1024 if sys.platform.startswith('linux') else 1024*1024))
        except ImportError: result['peak_rss_mib']=None
        if torch.cuda.is_available() and self.config.torch_device=='cuda':
            result['peak_cuda_reserved_mib']=float(torch.cuda.max_memory_reserved()/1024**2)
        return result

    def _log(self,row):
        with (self.output_dir/'metrics.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        if self.tensorboard is not None:
            for key,value in row.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value):
                    self.tensorboard.add_scalar(key,value,global_step=len(self.history))
    def close(self):
        close=getattr(self._env,'close',None)
        if close is not None: close()
        self._env=None
        if self.tensorboard is not None: self.tensorboard.close()
        self.ledger.close()
