"""Learned action values on frozen-per-step world embeddings; DQN TD regression."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,latent_dim,lr=1e-3,device='cpu'):
        super().__init__(); self.net=nn.Sequential(nn.Linear(latent_dim,64),nn.ReLU(),nn.Linear(64,5)); self.to(device)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr); self.device_name=device; self.train_steps=0
    def select(self,world,obs,epsilon,rng):
        if rng.random()<epsilon: return int(rng.integers(5))
        with torch.no_grad():
            z=world.encoder(world._tensor(obs[None]))
            return int(self.net(z).argmax(-1).item())
    def train_batch(self,world,experiences,gamma=.95):
        import numpy as np
        obs=np.stack([x.observation for x in experiences]); nxt=np.stack([x.next_observation for x in experiences])
        actions=torch.tensor([x.action for x in experiences],dtype=torch.long,device=self.device_name)
        rewards=torch.tensor([x.reward for x in experiences],dtype=torch.float32,device=self.device_name)
        terminal=torch.tensor([x.terminated for x in experiences],dtype=torch.float32,device=self.device_name)
        with torch.no_grad():
            z=world.encoder(world._tensor(obs)); nz=world.encoder(world._tensor(nxt))
            target=rewards+gamma*(1-terminal)*self.net(nz).max(-1).values
        self.optimizer.zero_grad(set_to_none=True)
        loss=F.smooth_l1_loss(self.net(z).gather(1,actions[:,None]).squeeze(1),target)
        if not torch.isfinite(loss): raise FloatingPointError('Actor loss nonfinite')
        loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),1.0)
        self.optimizer.step(); self.train_steps+=1; return float(loss.detach())
    def save(self,path):
        """Standalone policy artifact; pair with its corresponding world encoder."""
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'schema_version':1,'state_dict':self.state_dict(),
                    'optimizer':self.optimizer.state_dict(),'train_steps':self.train_steps},path)
    def restore(self,path):
        state=torch.load(path,map_location=self.device_name,weights_only=True)
        if state.get('schema_version')!=1 or type(state.get('train_steps')) is not int or state['train_steps']<0:
            raise ValueError('Invalid actor checkpoint')
        self.load_state_dict(state['state_dict'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.train_steps=state['train_steps']
        return self.train_steps
