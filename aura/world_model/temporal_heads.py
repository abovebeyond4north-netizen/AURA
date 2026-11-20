"""Action-sequence-conditioned direct prediction at multiple temporal scales.

Separate heads implement explicit h=1,2,4 predictors. They share the trained
visual encoder but are not claimed to be abstract symbolic concepts. Training
requires contiguous *observed* trajectories and never crosses terminals.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TemporalHeads(nn.Module):
    def __init__(self, encoder, latent_dim: int, horizons=(1,2,4),hidden_dim=64, lr=.001,device='cpu',freeze_encoder=False,mode='absolute'):
        super().__init__()
        self.encoder=encoder
        self.freeze_encoder=bool(freeze_encoder)
        if mode not in ('absolute','residual'):
            raise ValueError('Prediction mode must be absolute or residual')
        self.mode=mode
        self.horizons=tuple(sorted(set(map(int,horizons))))
        if not self.horizons or any(x<1 or x>32 for x in self.horizons):
            raise ValueError('Unsupported temporal horizon')
        self.heads=nn.ModuleDict({str(h):nn.Sequential(
            nn.Linear(latent_dim+5*h,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,3*7*7),nn.Sigmoid()) for h in self.horizons})
        # Behavioral/abstract head: predict the effect of an entire action
        # sequence on the *learned latent state*, not on a simulator label.
        self.latent_heads=nn.ModuleDict({str(h):nn.Sequential(
            nn.Linear(latent_dim+5*h,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,latent_dim)) for h in self.horizons})
        self.latent_is_trained=False
        if self.mode=='residual':
            # The initial predictor copies the observation: a defensible
            # persistence baseline before any self-supervised updates.
            for head in self.heads.values():
                nn.init.zeros_(head[-2].weight)
                nn.init.zeros_(head[-2].bias)
            for head in self.latent_heads.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.device_name=device
        self.to(device)
        parameters=(list(self.heads.parameters())+list(self.latent_heads.parameters())
                    if self.freeze_encoder else list(self.parameters()))
        self.optimizer=torch.optim.Adam(parameters,lr=lr)

    def predict(self,observation,actions):
        h=len(actions)
        if h not in self.horizons: raise ValueError('No head trained for requested horizon')
        x=torch.as_tensor(np.asarray(observation),dtype=torch.float32,device=self.device_name)
        if x.ndim==3: x=x[None]
        a=torch.as_tensor(actions,dtype=torch.long,device=self.device_name)
        if a.ndim!=1 or len(a)!=h or torch.any((a<0)|(a>4)):
            raise ValueError('Invalid action sequence')
        if self.freeze_encoder:
            with torch.no_grad(): z=self.encoder(x)
        else: z=self.encoder(x)
        action_features=F.one_hot(a,num_classes=5).float().flatten()[None].expand(z.shape[0],-1)
        head_output=self.heads[str(h)](torch.cat([z,action_features],dim=-1)).view(-1,3,7,7)
        if self.mode=='residual':
            # Predict observable action-conditioned change, rather than
            # relearn the entire static image. Exact zero delta initially.
            return torch.clamp(x/255. + 2.0*(head_output-.5),0.0,1.0)
        return head_output

    def predict_latent(self,observation,actions):
        """Action-conditioned multi-step latent effect forecast (not observed evidence)."""
        h=len(actions)
        if h not in self.horizons: raise ValueError('No trained horizon for sequence')
        x=torch.as_tensor(np.asarray(observation),dtype=torch.float32,device=self.device_name)
        if x.ndim==3: x=x[None]
        a=torch.as_tensor(actions,dtype=torch.long,device=self.device_name)
        if a.ndim!=1 or torch.any((a<0)|(a>4)):
            raise ValueError('Invalid action sequence')
        if self.freeze_encoder:
            with torch.no_grad(): z=self.encoder(x)
        else: z=self.encoder(x)
        features=F.one_hot(a,num_classes=5).float().flatten()[None].expand(z.shape[0],-1)
        latent=self.latent_heads[str(h)](torch.cat([z,features],dim=-1))
        return z+latent if self.mode=='residual' else latent

    def evaluate_latent(self,trajectories):
        """Held-out learned-representation MSE at each observed horizon."""
        self.eval()
        result={}
        with torch.no_grad():
            for h in self.horizons:
                losses=[]
                for tr in trajectories:
                    for x,a,y in self._windows(tr,h):
                        predicted=self.predict_latent(x,a)
                        target=self.encoder(torch.as_tensor(y,dtype=torch.float32,
                                                  device=self.device_name)[None])
                        losses.append(float(F.mse_loss(predicted,target)))
                result[h]=float(np.mean(losses)) if losses else None
        return result

    def save(self,path,update_steps=0):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':self.heads.state_dict(),
                    'latent_state_dict':self.latent_heads.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),
                    'latent_is_trained':self.latent_is_trained,
                    'update_steps':int(update_steps),
                    'schema_version':3,'horizons':self.horizons,'mode':self.mode},p)

    def restore(self,path):
        data=torch.load(path,map_location=self.device_name,weights_only=True)
        # v1 heads predicted absolute images; never silently interpret a
        # v1 checkpoint as residual predictions.
        checkpoint_mode=data.get('mode','absolute')
        if checkpoint_mode!=self.mode or tuple(data['horizons'])!=self.horizons:
            raise ValueError('Incompatible temporal checkpoint mode or horizons')
        self.heads.load_state_dict(data['state_dict'],strict=True)
        # Schema 1/2 weights load without silently claiming their new latent
        # heads were trained. Both legacy and new checkpoints are supported.
        if 'latent_state_dict' in data:
            self.latent_heads.load_state_dict(data['latent_state_dict'],strict=True)
            self.latent_is_trained=bool(data.get('latent_is_trained',False))
        else:
            self.latent_is_trained=False
        if 'optimizer_state_dict' in data:
            self.optimizer.load_state_dict(data['optimizer_state_dict'])
        return int(data['update_steps'])

    @staticmethod
    def _windows(trajectory,h):
        """Trajectory consists of (images[T+1], actions[T], done_or_truncated[T])."""
        observations, actions, endings=trajectory
        if len(observations)!=len(actions)+1 or len(endings)!=len(actions):
            raise ValueError('Broken trajectory contract')
        for t in range(len(actions)-h+1):
            # An endpoint terminal at t+h-1 is valid; earlier terminal isn't.
            if any(endings[t:t+h-1]): continue
            yield observations[t],actions[t:t+h],observations[t+h]

    def evaluate(self,trajectories):
        self.eval()
        result={}
        with torch.no_grad():
            for h in self.horizons:
                losses=[]
                for trajectory in trajectories:
                    for x,a,y in self._windows(trajectory,h):
                        target=torch.as_tensor(y,dtype=torch.float32,device=self.device_name)[None]/255.
                        losses.append(float(F.mse_loss(self.predict(x,a),target)))
                result[h]=float(np.mean(losses)) if losses else None
        return result

    def train_trajectories(self,trajectories,updates=10):
        examples={h:[w for traj in trajectories for w in self._windows(traj,h)] for h in self.horizons}
        active=[h for h in self.horizons if examples[h]]
        if not active: return {'updates':0,'reason':'no contiguous observed sequences'}
        if updates<0 or updates>10000: raise ValueError('Invalid update count')
        self.train()
        rng=np.random.default_rng(904)
        last=0.
        for step in range(updates):
            h=active[step%len(active)]
            w=examples[h][int(rng.integers(len(examples[h])))]
            x,a,y=w
            target=torch.as_tensor(y,dtype=torch.float32,device=self.device_name)[None]/255.
            self.optimizer.zero_grad(set_to_none=True)
            pred=self.predict(x,a)
            target_z=self.encoder(torch.as_tensor(y,dtype=torch.float32,
                              device=self.device_name)[None]).detach()
            pred_z=self.predict_latent(x,a)
            loss=F.mse_loss(pred,target)+0.02*F.mse_loss(pred_z,target_z)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite temporal-head loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step()
            self.latent_is_trained=True
            last=float(loss.detach())
        return {'updates':updates,'last_loss':last,'available_horizons':active}
