"""Action-conditioned recurrent latent ensemble trained on image prediction.
Output includes next-image regression, ensuring gradients reach the encoder even
if the latent matching objective could be minimized by feature collapse.
"""
from __future__ import annotations
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from aura.perception.encoder import ObservationEncoder

class DynamicsHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_actions: int = 5):
        super().__init__()
        self.cell = nn.GRUCell(latent_dim+n_actions, hidden_dim)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.image = nn.Sequential(nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 3*7*7), nn.Sigmoid())
        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor | None = None):
        a = F.one_hot(actions.long(), num_classes=5).float()
        h = self.cell(torch.cat([z,a],dim=-1), hidden)
        return self.latent(h), self.image(h).view(-1,3,7,7), h

class WorldModel(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, ensemble_size=3, lr=1e-3, device='cpu'):
        super().__init__()
        self.encoder = ObservationEncoder(latent_dim)
        self.heads = nn.ModuleList([DynamicsHead(latent_dim,hidden_dim) for _ in range(ensemble_size)])
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 3*7*7), nn.Sigmoid())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        self.device_name = device
        self.train_steps = 0
        self._seen_latents = []

    def _tensor(self, x): return torch.as_tensor(x, device=self.device_name, dtype=torch.float32)

    def prediction(self, obs, actions):
        z = self.encoder(self._tensor(obs))
        return [head(z, torch.as_tensor(actions, device=self.device_name))[1] for head in self.heads]

    @torch.no_grad()
    def prediction_loss(self, batch) -> float:
        self.eval()
        x, a, y = batch
        y = self._tensor(y) / 255.
        p = torch.stack(self.prediction(x,a)).mean(0)
        return float(F.mse_loss(p,y))

    def train_batch(self, batch):
        self.train()
        x,a,y = batch
        x,y = self._tensor(x)/255., self._tensor(y)/255.
        acts = torch.as_tensor(a,device=self.device_name,dtype=torch.long)
        self.optimizer.zero_grad(set_to_none=True)
        z = self.encoder(x)
        with torch.no_grad(): target_z = self.encoder(y)
        raw = torch.stack([head(z,acts)[1] for head in self.heads])
        pz = torch.stack([head(z,acts)[0] for head in self.heads])
        recon = self.decoder(z).view(-1,3,7,7)
        image_loss = F.mse_loss(raw, y.unsqueeze(0).expand_as(raw))
        latent_loss = F.mse_loss(pz,target_z.unsqueeze(0).expand_as(pz))
        reconstruct = F.mse_loss(recon,x)
        # Penalizes constant embeddings when batch includes sufficient diversity.
        std = torch.sqrt(z.var(0, unbiased=False)+1e-4)
        variance_loss = F.relu(0.3-std).mean()
        loss = image_loss + 0.05*latent_loss + 0.25*reconstruct + 0.05*variance_loss
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite model loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(),1.0,error_if_nonfinite=True)
        self.optimizer.step()
        self.train_steps += 1
        return {'loss': float(loss.detach()), 'image_loss': float(image_loss.detach()),
                'variance': float(std.mean().detach()), 'embedding_collapse_dims': int((std<0.02).sum())}

    @torch.no_grad()
    def estimate_uncertainty(self, obs, actions) -> torch.Tensor:
        z = self.encoder(self._tensor(obs))
        heads = torch.stack([h(z,torch.as_tensor(actions,device=self.device_name))[0] for h in self.heads])
        return heads.var(0,unbiased=False).mean(-1)

    @torch.no_grad()
    def imagine(self, obs, actions: list[int]):
        """IMAGINED latent trajectories; MUST NOT be added to observed replay."""
        z = self.encoder(self._tensor(obs))
        hidden = [None] * len(self.heads)
        trajectory = []
        for action in actions:
            predictions = [h(z,torch.full((z.shape[0],),action,device=z.device),hidden[i]) for i,h in enumerate(self.heads)]
            z = torch.stack([p[0] for p in predictions]).mean(0)
            hidden = [p[2] for p in predictions]
            trajectory.append(z.detach().clone())
        return trajectory

    def save(self, path: str | Path, metadata: dict | None = None):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':self.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'train_steps':self.train_steps,'metadata':metadata or {}},p)

    def restore(self, path: str | Path):
        # weights_only blocks arbitrary pickle globals on loading.
        data = torch.load(path, map_location=self.device_name, weights_only=True)
        self.load_state_dict(data['state_dict'], strict=True)
        self.optimizer.load_state_dict(data['optimizer'])
        self.train_steps = int(data['train_steps'])
        return data['metadata']
