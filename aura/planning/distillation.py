"""Opt-in neural imitation of observed rewarded episodes; not a promotion gate."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from aura.memory.episodic import EpisodicMemory


def rewarded_episodes(ledger):
    """Only complete, chained, positive-terminal observed experience is eligible."""
    if not ledger.verify():
        raise ValueError('Invalid evidence ledger')
    grouped=defaultdict(list)
    memory=EpisodicMemory(ledger)
    for record in ledger.records('experience'):
        exp=memory.reconstruct(record)
        grouped[exp.episode_id].append(exp)
    eligible=[]
    for episode in grouped.values():
        final=episode[-1]
        if not final.terminated or final.truncated or final.reward<=0:
            continue
        if (any(e.terminated or e.truncated for e in episode[:-1]) or
            any(e.seed!=final.seed or e.context!=final.context for e in episode) or
            any(not np.isfinite(e.reward) for e in episode) or
            any(not np.array_equal(a.next_observation,b.observation) for a,b in zip(episode,episode[1:]))):
            raise ValueError('Inconsistent rewarded episode evidence')
        eligible.append(tuple(episode))
    return eligible


def split_episodes(episodes, seed=0):
    """Keep entire environment-seed groups together, including repeated episodes."""
    groups=sorted({(ep[0].context,ep[0].seed) for ep in episodes})
    if len(groups)<2: raise ValueError('At least two distinct environment seed groups required')
    rng=np.random.default_rng(seed)
    held={groups[int(i)] for i in rng.permutation(len(groups))[:max(1,len(groups)//4)]}
    return ([ep for ep in episodes if (ep[0].context,ep[0].seed) not in held],
            [ep for ep in episodes if (ep[0].context,ep[0].seed) in held])


class RecurrentImitation(nn.Module):
    """Fresh image encoder and recurrent policy; no privileged inputs or route lookup."""
    def __init__(self, hidden=64):
        super().__init__()
        if type(hidden) is not int or not 8<=hidden<=256: raise ValueError('Invalid hidden size')
        self.hidden=hidden
        self.image=nn.Sequential(nn.Flatten(),nn.Linear(147,hidden),nn.ReLU())
        self.gru=nn.GRU(hidden+6,hidden,batch_first=True)
        self.head=nn.Linear(hidden,5)
        self.reset_episode()

    def forward(self, images, previous, state=None):
        # Inputs are raw images and previous observed actions; 5 marks episode start.
        b,t=images.shape[:2]
        z=self.image(images.reshape(b*t,3,7,7).float()/255).reshape(b,t,self.hidden)
        z=torch.cat((z,F.one_hot(previous,num_classes=6).float()),dim=-1)
        out,state=self.gru(z,state)
        return self.head(out),state

    def reset_episode(self):
        self._state=None; self._previous=5

    @torch.no_grad()
    def choose(self, observation):
        image=torch.as_tensor(np.asarray(observation),device=self.head.weight.device).unsqueeze(0).unsqueeze(0)
        previous=torch.tensor([[self._previous]],device=image.device)
        logits,self._state=self(image,previous,self._state)
        self._previous=int(logits.argmax(-1).item())
        return self._previous

    def save(self,path):
        torch.save({'schema_version':1,'hidden':self.hidden,'weights':self.state_dict()},Path(path))

    @classmethod
    def load(cls,path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema_version')!=1: raise ValueError('Unknown imitation checkpoint')
        model=cls(payload['hidden']); model.load_state_dict(payload['weights'],strict=True); model.eval()
        return model


def _batch(episodes):
    if not episodes or not all(episodes): raise ValueError('Nonempty episodes required')
    t=max(map(len,episodes)); b=len(episodes)
    images=np.zeros((b,t,3,7,7),dtype=np.uint8); actions=np.full((b,t),-100,dtype=np.int64)
    previous=np.full((b,t),5,dtype=np.int64)
    for i,episode in enumerate(episodes):
        n=len(episode); images[i,:n]=np.stack([e.observation for e in episode]); actions[i,:n]=[e.action for e in episode]
        previous[i,1:n]=actions[i,:n-1]
    return tuple(torch.from_numpy(x) for x in (images,previous,actions))


def _loss_and_accuracy(model,batch):
    obs,prev,actions=batch; logits,_=model(obs,prev)
    loss=F.cross_entropy(logits.flatten(0,1),actions.flatten(),ignore_index=-100)
    valid=actions!=-100
    return loss,float((logits.argmax(-1)[valid]==actions[valid]).float().mean().detach())


def budgeted_episode_indices(lengths, fresh_count, step):
    """Rotate whole sequences within both fresh-only token and padded-cell caps.

    Always present a fresh episode; old episodes compete for remaining slots.
    Deterministic rotation is a budgeted baseline, not learned prioritization.
    """
    if (type(fresh_count) is not int or not 1 <= fresh_count <= len(lengths)
            or type(step) is not int or step < 0 or any(type(n) is not int or n < 1 for n in lengths)):
        raise ValueError('Invalid fresh episode budget')
    if fresh_count == len(lengths):
        return list(range(fresh_count))
    fresh = list(range(fresh_count))
    fresh = fresh[step % fresh_count:] + fresh[:step % fresh_count]
    old = list(range(fresh_count, len(lengths)))
    offset = step % len(old)
    old = old[offset:] + old[:offset]
    transition_cap = sum(lengths[:fresh_count])
    padded_cap = fresh_count * max(lengths[:fresh_count])
    selected = [fresh[0]]
    # Interleave old and new episodes where both fit, never split recurrent sequences.
    candidates = []
    for i in range(max(len(old), len(fresh)-1)):
        if i < len(old): candidates.append(old[i])
        if i + 1 < len(fresh): candidates.append(fresh[i+1])
    for i in candidates:
        proposal = selected + [i]
        if (sum(lengths[j] for j in proposal) <= transition_cap
                and len(proposal) * max(lengths[j] for j in proposal) <= padded_cap):
            selected = proposal
    return selected


def fit_imitation(model, training, validation, *, updates=120, lr=.003, budgeted_fresh_count=None):
    """Fixed update count; validation is measured but never selects weights."""
    if type(updates) is not int or not 1<=updates<=1000 or not np.isfinite(lr) or not 0<lr<=.1:
        raise ValueError('Invalid optimization budget')
    train_groups={(ep[0].context,ep[0].seed) for ep in training}
    val_groups={(ep[0].context,ep[0].seed) for ep in validation}
    train_ids={e.evidence_id for ep in training for e in ep}
    val_ids={e.evidence_id for ep in validation for e in ep}
    if train_groups&val_groups or train_ids&val_ids: raise ValueError('Training and validation overlap')
    lengths = [len(ep) for ep in training]
    if budgeted_fresh_count is not None:
        budgeted_episode_indices(lengths, budgeted_fresh_count, 0)
    a,b=_batch(training),_batch(validation)
    def measure():
        with torch.no_grad():
            lt,at=_loss_and_accuracy(model,a); lv,av=_loss_and_accuracy(model,b)
            return {'train_loss':float(lt),'train_accuracy':at,'validation_loss':float(lv),'validation_accuracy':av}
    before=measure(); opt=torch.optim.Adam(model.parameters(),lr=lr)
    model.train()
    presentations = [0] * len(training)
    transition_work = padded_work = 0
    for step in range(updates):
        indices = (list(range(len(training))) if budgeted_fresh_count is None else
                   budgeted_episode_indices(lengths, budgeted_fresh_count, step))
        batch = a if len(indices) == len(training) and indices == list(range(len(training))) else _batch([training[i] for i in indices])
        for i in indices: presentations[i] += 1
        transition_work += sum(lengths[i] for i in indices)
        padded_work += len(indices) * max(lengths[i] for i in indices)
        opt.zero_grad(set_to_none=True); loss,_=_loss_and_accuracy(model,batch)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite imitation loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    model.eval()
    return {'before':before,'after':measure(),'updates':updates,
            'training_transitions':int((a[2]!=-100).sum()),'validation_transitions':int((b[2]!=-100).sum()),
            'training_transition_presentations':transition_work,
            'training_padded_step_presentations':padded_work,
            'episode_presentations':presentations,
            'fresh_transition_cap_per_update':(sum(lengths[:budgeted_fresh_count]) if budgeted_fresh_count is not None else None),
            'fresh_padded_cap_per_update':(budgeted_fresh_count * max(lengths[:budgeted_fresh_count]) if budgeted_fresh_count is not None else None),
            'rehearsal_transition_presentations':(sum(n*k for n,k in zip(lengths[budgeted_fresh_count:],presentations[budgeted_fresh_count:])) if budgeted_fresh_count is not None else None),
            'measurement_padded_step_presentations':2*(a[0].shape[0]*a[0].shape[1]+b[0].shape[0]*b[0].shape[1])}
