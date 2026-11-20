"""Horizon-specific evaluation with recurrent imagined rollouts; no claim of abstraction."""
import numpy as np
import torch

def horizon_errors(world, trajectories, horizons=(1,2,4)):
    """Input: list of (observations[T+1,3,7,7], actions[T])."""
    errors = {int(h):[] for h in horizons}
    for observations, actions in trajectories:
        for h in horizons:
            if len(actions)<h: continue
            for t in range(len(actions)-h+1):
                imag=world.imagine(observations[t:t+1],list(actions[t:t+h]))[-1]
                with torch.no_grad(): truth=world.encoder(world._tensor(observations[t+h:t+h+1]))
                errors[h].append(float(torch.nn.functional.mse_loss(imag,truth)))
    return {h:float(np.mean(vals)) if vals else float('nan') for h,vals in errors.items()}
