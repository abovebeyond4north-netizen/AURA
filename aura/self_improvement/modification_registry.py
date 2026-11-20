"""Strict typed whitelist; cannot edit code, verifier, paths or acceptance rules."""
from aura.types import Modification
from aura.config import Config

PERMITTED = {
    'learning_rate': ('lr','multiply',0.00001,0.02),
    'exploration_rate': ('epsilon','set',0.01,0.9),
    'latent_capacity': ('latent_dim','set',4,128),
    'model_capacity': ('hidden_dim','set',8,256),
}

def apply_modification(config: Config,mod: Modification) -> Config:
    if mod.target not in PERMITTED or list(mod.parameters)!=['value']:
        raise PermissionError('Modification outside the permitted registry')
    field,operation,minimum,maximum=PERMITTED[mod.target]
    if operation!=mod.operation: raise PermissionError('Unregistered operation')
    value=mod.parameters['value']
    if not isinstance(value,(int,float)) or isinstance(value,bool): raise ValueError('Invalid numerical modification')
    new=config.model_copy(deep=True)
    proposed=getattr(config,field)*value if operation=='multiply' else value
    if not minimum<=proposed<=maximum: raise ValueError('Candidate out of configured range')
    if field in ('latent_dim','hidden_dim'):
        if int(proposed)!=proposed: raise ValueError('Dimensions must be integral')
        proposed=int(proposed)
    setattr(new,field,proposed)
    return new
