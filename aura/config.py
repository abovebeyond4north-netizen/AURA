"""Validated and reproducible configuration; no arbitrary executable values."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import random
import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field, ConfigDict

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    seed: int = Field(default=42, ge=0)
    device: Literal['auto', 'cpu', 'cuda'] = 'cpu'
    steps: int = Field(default=800, ge=1, le=10_000_000)
    batch_size: int = Field(default=32, ge=2, le=4096)
    warmup: int = Field(default=32, ge=2)
    latent_dim: int = Field(default=16, ge=4, le=512)
    hidden_dim: int = Field(default=32, ge=8, le=1024)
    ensemble_size: int = Field(default=3, ge=2, le=20)
    lr: float = Field(default=0.001, gt=0, le=0.1)
    max_episode_steps: int = Field(default=40, ge=5, le=10000)
    epsilon: float = Field(default=0.2, ge=0, le=1)
    curiosity_policy: Literal['ensemble', 'validated_progress', 'visual_frontier', 'reward_routes'] = 'ensemble'
    log_dir: str = 'runs/default'
    evaluation_checkpoint: str | None = None
    improvement_interval: int = Field(default=0, ge=0)
    temporal_updates_per_episode: int = Field(default=8, ge=0, le=128)

    # Shadow candidates are off by default and never auto-activated.
    imitation_inherit: bool = False
    imitation_budgeted_rehearsal: bool = False
    imitation_rehearsal_episodes: int = Field(default=0, ge=0, le=16)
    imitation_interval: int = Field(default=0, ge=0, le=10000)
    imitation_updates: int = Field(default=120, ge=1, le=1000)
    imitation_max_candidates: int = Field(default=2, ge=1, le=10)
    imitation_eval_episodes: int = Field(default=3, ge=1, le=10)

    @property
    def torch_device(self) -> str:
        return 'cuda' if self.device == 'auto' and torch.cuda.is_available() else ('cpu' if self.device == 'auto' else self.device)

    @classmethod
    def read(cls, path: str | Path) -> 'Config':
        with Path(path).open() as f:
            return cls.model_validate(yaml.safe_load(f))

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
