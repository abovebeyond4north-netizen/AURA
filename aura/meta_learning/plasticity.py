"""Monitor dormant encoder channels and generate rejuvenated candidate copies.

A rejuvenated candidate must be evaluated by the independent verifier before
any active agent adopts it; this operator does not promote anything.
"""
from __future__ import annotations
import torch
from aura.world_model.dynamics import WorldModel

class PlasticityMonitor:
    def __init__(self,threshold=0.01): self.threshold=threshold
    @torch.no_grad()
    def measure(self,model:WorldModel,observations):
        z=model.encoder(model._tensor(observations))
        std=z.std(0,unbiased=False)
        inactive=(std<self.threshold).nonzero(as_tuple=False).flatten().tolist()
        return {'inactive_indices':inactive,'inactive_fraction':len(inactive)/z.shape[-1],
                'mean_latent_std':float(std.mean())}

    @torch.no_grad()
    def candidate_rejuvenation(self,model:WorldModel,observations,max_fraction=.2):
        if not 0<=max_fraction<=1: raise ValueError('Fraction out of range')
        stats=self.measure(model,observations)
        n=min(len(stats['inactive_indices']),int(model.encoder.net[5].out_features*max_fraction))
        # Make a separate, same-shaped model; never modify active model weights.
        original_lr=model.optimizer.param_groups[0]['lr']
        candidate=WorldModel(model.encoder.net[5].out_features,model.heads[0].hidden_dim,
                              len(model.heads),original_lr,model.device_name)
        candidate.load_state_dict(model.state_dict())
        layer=candidate.encoder.net[5]
        for index in stats['inactive_indices'][:n]:
            torch.nn.init.normal_(layer.weight[index:index+1],std=.05)
            layer.bias[index].zero_()
        return candidate,{'rejuvenated_indices':stats['inactive_indices'][:n],
                          'requires_independent_promotion':True}
