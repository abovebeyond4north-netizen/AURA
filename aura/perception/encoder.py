"""Randomly initialized visual encoder for 3x7x7 uint8 observations."""
import torch
from torch import nn

class ObservationEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(32*7*7, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4 or observations.shape[1:] != (3,7,7):
            raise ValueError('Expected [batch,3,7,7] images')
        return self.net(observations.float() / 255.0 if observations.max().item() > 1 else observations.float())

    @staticmethod
    def statistics(z: torch.Tensor) -> dict[str,float]:
        return {'mean_std': float(z.std(dim=0, unbiased=False).mean().detach()),
                'collapsed_dims': int((z.std(dim=0, unbiased=False) < 1e-3).sum().detach())}
