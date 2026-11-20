#!/usr/bin/env python3
"""Evaluate learned checkpoint on independent development seed distribution."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from aura.config import Config
from aura.world_model.dynamics import WorldModel
from aura.world_model.temporal_heads import TemporalHeads
import torch
from aura.evaluation.runner import evaluate_world_model,evaluate_temporal_heads,evaluate_temporal_latent_heads

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evaluation.yaml');p.add_argument('--checkpoint',default=None);p.add_argument('--temporal-checkpoint',default=None)
    args=p.parse_args();c=Config.read(args.config)
    path=Path(args.checkpoint or c.evaluation_checkpoint or Path(c.log_dir)/'checkpoint.pt')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if 'config' not in payload.get('metadata',{}): raise ValueError('Checkpoint missing architecture and training metadata')
    stored=Config.model_validate(payload['metadata']['config'])
    world=WorldModel(stored.latent_dim,stored.hidden_dim,stored.ensemble_size,stored.lr,c.torch_device)
    meta=world.restore(path)
    report={task:evaluate_world_model(world,task,i) for i,task in enumerate(['key','switch','hidden','altered','noise'])}
    temporal_checkpoint=(Path(args.temporal_checkpoint) if args.temporal_checkpoint
                         else (path.with_name('latest_temporal_checkpoint.pt')
                               if path.name=='latest_world_checkpoint.pt'
                               else path.with_name('temporal.pt')))
    if args.temporal_checkpoint is not None and not temporal_checkpoint.is_file():
        raise FileNotFoundError(f'Explicit temporal checkpoint is missing: {temporal_checkpoint}')
    if temporal_checkpoint.is_file():
        saved=torch.load(temporal_checkpoint,map_location='cpu',weights_only=True)
        temporal=TemporalHeads(world.encoder,stored.latent_dim,(1,2,4),
                               hidden_dim=stored.hidden_dim,lr=stored.lr,
                               device=c.torch_device,freeze_encoder=True,
                               mode=saved.get('mode','absolute'))
        restored_updates=temporal.restore(temporal_checkpoint)
        for i,task in enumerate(['key','switch','hidden','altered','noise']):
            report[task].update(evaluate_temporal_heads(temporal,task,i))
            if temporal.latent_is_trained:
                report[task].update(evaluate_temporal_latent_heads(temporal,task,i))
        report['_temporal']={'checkpoint':str(temporal_checkpoint),
                             'updates':restored_updates,'mode':temporal.mode,
                             'latent_is_trained':temporal.latent_is_trained,
                             'schema_version':saved.get('schema_version',1)}
    report['_metadata']={'checkpoint':str(path),'checkpoint_training_seed':stored.seed,'evaluation_seed':c.seed}
    out=Path(c.log_dir)/'development_evaluation.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
